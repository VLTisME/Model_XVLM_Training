"""Train STAR-v3.

Usage:
    python scripts/train.py --config configs/star_v3_100k.yaml
    python scripts/train.py --config configs/star_v3_100k.yaml --overfit-one-batch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# allow `python scripts/train.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import load_config, parse_overrides  # noqa: E402
from star.data import GroupedBatchSampler, PABDataset, PairBatchSampler, collate_fn  # noqa: E402
from star.engine import Trainer                  # noqa: E402
from star.models import STARModel                # noqa: E402
from star.utils import get_logger, seed_everything  # noqa: E402

log = get_logger("star.train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overfit-one-batch", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. optim.lr_lora=1e-4")
    ap.add_argument("--resume", default=None, help="path to last.pth to continue a run across commits")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop + save last.pth after N hours (keeps a Kaggle commit under the 9h limit)")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")

    model = STARModel(cfg)
    tokenizer = model.backbone.tokenizer

    train_ds = PABDataset(
        cfg.data.manifest, cfg.data.image_root, tokenizer, split="train",
        image_size=cfg.data.image_size, max_token=cfg.data.max_token, train=True,
        lhp_kwargs={"min_scale": cfg.data.lhp_min_scale, "use_bbox": cfg.data.lhp_use_bbox,
                    "enabled": cfg.data.lhp_enabled},
    )
    val_ds = PABDataset(
        cfg.data.manifest, cfg.data.image_root, tokenizer, split="valb",
        image_size=cfg.data.image_size, max_token=cfg.data.max_token, train=False,
    )

    if cfg.data.group_by == "pair":
        pairs, groups = train_ds.pairs()
        assert pairs, ("group_by=pair requires manifest columns image_id + pair_image_id "
                       "(anchor rows carry the mined hard image id)")
        sampler = PairBatchSampler(pairs, groups, cfg.train.batch_size, seed=cfg.train.seed)
        log.info(f"PairBatchSampler: {len(pairs)} pairs -> {len(sampler)} batches/epoch "
                 f"({cfg.train.batch_size // 2} video-distinct pairs/batch)")
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.data.num_workers,
                                  collate_fn=collate_fn, pin_memory=True)
    elif cfg.data.group_by != "none":
        sampler = GroupedBatchSampler(train_ds.group_ids(cfg.data.group_by), cfg.train.batch_size,
                                      cfg.data.group_fraction, seed=cfg.train.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg.data.num_workers,
                                  collate_fn=collate_fn, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                                  num_workers=cfg.data.num_workers, collate_fn=collate_fn,
                                  pin_memory=True, drop_last=True)

    trainer = Trainer(model, cfg, train_loader, val_ds, device)
    if args.max_hours:
        trainer.max_seconds = args.max_hours * 3600
    if args.overfit_one_batch:
        trainer.overfit_one_batch()
    else:
        if args.resume:
            trainer.resume_from(args.resume)
        trainer.train()


if __name__ == "__main__":
    main()
