"""Run the STAR-v3 inference pipeline on an eval manifest.

    python scripts/run_inference.py --ckpt best.pth --manifest eval.parquet \
        --image-root /data --base-ckpt xvlm_16m_base.th --out-dir infer_out --topk 100

Pipeline (per inference2.svg, trimmed): cosine Stage-1 -> Top-K -> cross-encoder ITM
re-rank (BLIP-style logit+cosine) -> Gale-Shapley rank-1 assignment -> top-10/query.
Reports metrics at every stage when the manifest queries carry ground truth (each query
row's own image is its GT). Writes metrics.json + answer.txt (+ ranks.pt).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import _merge, load_config, parse_overrides  # noqa: E402
from star.data import PABDataset                              # noqa: E402
from star.inference import run_pipeline                       # noqa: E402
from star.models import STARModel                             # noqa: E402
from star.utils import get_logger                             # noqa: E402

log = get_logger("star.infer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="trained best.pth")
    ap.add_argument("--manifest", required=True, help="eval parquet (valb rows; empty caption = distractor)")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--base-ckpt", default=None, help="X-VLM base weights (model construction)")
    ap.add_argument("--config", default="configs/star_v3_10k_kaggle.yaml")
    ap.add_argument("--out-dir", default="infer_out")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-gale-shapley", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")

    cfg = load_config(args.config, parse_overrides(args.set))
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    embedded = (raw.get("extra") or {}).get("cfg") or {}
    if "model" in embedded:
        _merge(cfg.model, embedded["model"])     # rebuild the EXACT trained architecture
    if args.base_ckpt:
        cfg.model.checkpoint = args.base_ckpt

    model = STARModel(cfg).to(device).eval()
    msg = model.load_state_dict(raw["model"], strict=False)
    log.info(f"ckpt loaded: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} "
             f"| pose_enabled={cfg.model.pose_enabled} | train-best mAP={raw.get('best_metric')}")
    del raw

    ds = PABDataset(args.manifest, args.image_root, model.backbone.tokenizer, split="valb",
                    image_size=cfg.data.image_size, max_token=cfg.data.max_token, train=False)
    log.info(f"eval rows={len(ds)}")

    t0 = time.time()
    res = run_pipeline(model, ds, device, topk=args.topk, batch_size=args.batch_size,
                       num_workers=args.num_workers, use_gale_shapley=not args.no_gale_shapley)
    mins = (time.time() - t0) / 60

    log.info(f"gallery={res['gallery_size']} queries={res['num_queries']} K={res['topk']} ({mins:.1f} min)")
    for stage in ("stage1", "rerank", "gale_shapley"):
        log.info(f"[{stage:12s}] " + " ".join(f"{k}={v:.4f}" for k, v in res[stage].items()))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {s: res[s] for s in ("stage1", "rerank", "gale_shapley")} |
        {"gallery_size": res["gallery_size"], "num_queries": res["num_queries"],
         "topk": res["topk"], "minutes": round(mins, 1)}, indent=2), encoding="utf-8")
    with open(out / "answer.txt", "w", encoding="utf-8") as f:
        for qi, ids in enumerate(res["top10"]):
            f.write(" ".join([str(qi)] + [str(x) for x in ids]) + "\n")
    torch.save(res["ranks"], out / "ranks.pt")
    log.info(f"wrote {out}/metrics.json, answer.txt, ranks.pt")


if __name__ == "__main__":
    main()
