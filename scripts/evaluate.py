"""Evaluate a checkpoint on VAL-B (bi-encoder retrieval: mAP / MRR / R@K)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import load_config              # noqa: E402
from star.data import PABDataset                 # noqa: E402
from star.engine import evaluate_retrieval       # noqa: E402
from star.models import STARModel                # noqa: E402
from star.utils import get_logger, load_checkpoint  # noqa: E402

log = get_logger("star.eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="valb")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STARModel(cfg).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)

    ds = PABDataset(cfg.data.manifest, cfg.data.image_root, model.backbone.tokenizer,
                    split=args.split, image_size=cfg.data.image_size,
                    max_token=cfg.data.max_token, train=False)
    report = evaluate_retrieval(model, ds, device, num_workers=cfg.data.num_workers)
    log.info(f"[{args.split}] {report}")


if __name__ == "__main__":
    main()
