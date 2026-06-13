"""Train the pairwise (duo) comparator on top of a FROZEN trained checkpoint.

Reuses the mined hard pairs (anchor's GT image vs its hard-negative image) as supervision:
the comparator must prefer the GT given the anchor caption. Backbone is frozen -> only the
~1M-param head trains -> a few minutes. Output: a head .pt (state_dict + dim) loaded at inference.

    python scripts/train_pairwise.py --ckpt /kaggle/working/v6/best.pth \
        --manifest manifest_10k_hard.parquet --image-root <DATA_ROOT> \
        --base-ckpt xvlm_16m_base.th --out /kaggle/working/v6/pairwise_head.pt --epochs 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import _merge, load_config        # noqa: E402
from star.data import PABDataset                    # noqa: E402
from star.models import PairwiseHead, STARModel      # noqa: E402
from star.utils import get_logger                    # noqa: E402

log = get_logger("star.pairwise")


class PairFeatDS(Dataset):
    """Yields (input_ids, attention_mask, img_pos, img_neg) per mined (anchor, hard) pair."""

    def __init__(self, ds, pairs):
        self.ds, self.pairs = ds, pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j = self.pairs[k]
        a, b = self.ds[i], self.ds[j]
        return a["input_ids"], a["attention_mask"], a["image"], b["image"]


def _collate(batch):
    return (torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]), torch.stack([b[3] for b in batch]))


@torch.no_grad()
def _features(model, ids, mask, img_pos, img_neg, device, use_amp):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        txt_embeds, _ = model.backbone.encode_text(ids, mask)
        ie_pos, _ = model.backbone.encode_image(img_pos)
        ie_neg, _ = model.backbone.encode_image(img_neg)
        h_pos = model.backbone.cross_feature(ie_pos, txt_embeds, mask)
        h_neg = model.backbone.cross_feature(ie_neg, txt_embeds, mask)
    return h_pos.float(), h_neg.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained best.pth (backbone frozen here)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--config", default="configs/star_v3_10k_kaggle.yaml")
    ap.add_argument("--out", default="pairwise_head.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = "cuda" in device
    log.info(f"device={device}")

    cfg = load_config(args.config)
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    emb = (raw.get("extra") or {}).get("cfg") or {}
    if "model" in emb:
        _merge(cfg.model, emb["model"])
    if args.base_ckpt:
        cfg.model.checkpoint = args.base_ckpt
    model = STARModel(cfg).to(device).eval()
    msg = model.load_state_dict(raw["model"], strict=False)
    log.info(f"ckpt loaded: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    for p in model.parameters():
        p.requires_grad_(False)
    del raw

    ds = PABDataset(args.manifest, args.image_root, model.backbone.tokenizer, split="train",
                    image_size=cfg.data.image_size, max_token=cfg.data.max_token, train=False)
    pairs, _ = ds.pairs()
    assert pairs, "manifest has no usable (anchor, hard) pairs (need pair_image_id + image_id)"
    log.info(f"train pairs={len(pairs)}")
    loader = DataLoader(PairFeatDS(ds, pairs), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=_collate, drop_last=True)

    ids, mask, pos, neg = next(iter(loader))
    h_pos, _ = _features(model, ids.to(device), mask.to(device), pos.to(device), neg.to(device),
                         device, use_amp)
    dim = h_pos.size(-1)
    head = PairwiseHead(dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    bce = nn.BCEWithLogitsLoss()
    log.info(f"PairwiseHead dim={dim} hidden={args.hidden} | trainable={sum(p.numel() for p in head.parameters())/1e6:.2f}M")

    for ep in range(args.epochs):
        head.train()
        t0, tot, n, acc = time.time(), 0.0, 0, 0.0
        for ids, mask, pos, neg in loader:
            h_pos, h_neg = _features(model, ids.to(device), mask.to(device),
                                     pos.to(device), neg.to(device), device, use_amp)
            lp, ln = head(h_pos, h_neg), head(h_neg, h_pos)      # want lp>0, ln<0
            loss = bce(lp, torch.ones_like(lp)) + bce(ln, torch.zeros_like(ln))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            acc += 0.5 * ((lp > 0).float().mean() + (ln < 0).float().mean()).item()
            n += 1
        log.info(f"epoch {ep}: loss={tot/n:.4f} pair-acc={acc/n:.3f} ({(time.time()-t0)/60:.1f} min)")

    torch.save({"state_dict": head.state_dict(), "dim": dim, "hidden": args.hidden}, args.out)
    log.info(f"saved {args.out} (dim={dim})")


if __name__ == "__main__":
    main()
