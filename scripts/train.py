"""Train STAR-v3.

Usage:
    python scripts/train.py --config configs/star_v3_100k.yaml
    python scripts/train.py --config configs/star_v3_100k.yaml --overfit-one-batch
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# allow `python scripts/train.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import load_config, parse_overrides  # noqa: E402
from star.data import (  # noqa: E402
    GroupedBatchSampler,
    PABDataset,
    PairAwareCollator,
    PairBatchSampler,
    PairMixedBatchSampler,
    collate_fn,
)
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
    ap.add_argument("--init-from", default=None,
                    help="load model WEIGHTS ONLY from a checkpoint (fine-tune: fresh optimizer/"
                         "scheduler/step, new LR + loss weights). Use this — not --resume — to "
                         "continue from best.pth with a CHANGED recipe (e.g. higher lambda_itm).")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop + save last.pth after N hours (keeps a Kaggle commit under the 9h limit)")
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    log.info(f"device={device}")

    model = STARModel(cfg)
    init_checkpoint = None
    if args.init_from:                                  # fine-tune: nap weight, KHONG nap optimizer/step
        from star.utils.checkpoint import load_checkpoint
        init_checkpoint = load_checkpoint(
            args.init_from, model, restore_rng=False
        )  # weights only; keep the new run's seeded RNG
        log.info(f"[init-from] loaded weights from {args.init_from} "
                 f"(best_metric was {init_checkpoint.get('best_metric')}) "
                 "-> fresh optimizer/scheduler/step")
    tokenizer = model.backbone.tokenizer

    aug_kwargs = {
        "min_scale": cfg.data.lhp_min_scale,
        "use_bbox": cfg.data.lhp_use_bbox,
        "enabled": cfg.data.lhp_enabled,
        "motion_blur_p": cfg.data.motion_blur_p,
        "jpeg_p": cfg.data.jpeg_p,
        "downscale_p": cfg.data.downscale_p,
        "color_jitter_p": cfg.data.color_jitter_p,
        "noise_p": cfg.data.noise_p,
        "erase_p": cfg.data.erase_p,
        "max_ops": cfg.data.max_aug_ops,
    }
    paired_training = cfg.data.group_by in {"pair", "pair_mixed"} and cfg.data.pair_consistent_aug
    train_ds = PABDataset(
        cfg.data.manifest, cfg.data.image_root, tokenizer, split="train",
        image_size=cfg.data.image_size, max_token=cfg.data.max_token, train=True,
        lhp_kwargs=aug_kwargs,
        defer_transform=paired_training,
    )

    loader_kwargs = {
        "num_workers": cfg.data.num_workers,
        "pin_memory": True,
        "persistent_workers": cfg.data.persistent_workers and cfg.data.num_workers > 0,
    }
    if cfg.data.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.data.prefetch_factor
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
        pair_collate = PairAwareCollator(train_ds.transform, hard_pairs=None)
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler, collate_fn=pair_collate, **loader_kwargs
        )
    elif cfg.data.group_by == "pair_mixed":
        pairs, groups = train_ds.pairs()
        assert pairs, ("group_by=pair_mixed requires manifest columns image_id + pair_image_id "
                       "(anchor rows carry the mined hard image id)")
        hard_pairs = (
            cfg.train.batch_size // 2
            if cfg.data.pair_hard_pairs < 0
            else cfg.data.pair_hard_pairs
        )
        sampler = PairMixedBatchSampler(
            pairs,
            groups,
            cfg.train.batch_size,
            hard_pairs=hard_pairs,
            num_samples=len(train_ds),
            seed=cfg.train.seed,
        )
        log.info(f"PairMixedBatchSampler: {len(pairs)} pairs -> {len(sampler)} batches/epoch "
                 f"({hard_pairs} hard pairs/batch, "
                 f"{cfg.train.batch_size - 2 * hard_pairs} random fillers/batch)")
        pair_collate = PairAwareCollator(train_ds.transform, hard_pairs=hard_pairs)
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler, collate_fn=pair_collate, **loader_kwargs
        )
    elif cfg.data.group_by != "none":
        sampler = GroupedBatchSampler(train_ds.group_ids(cfg.data.group_by), cfg.train.batch_size,
                                      cfg.data.group_fraction, seed=cfg.train.seed)
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler, collate_fn=collate_fn, **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            **loader_kwargs,
        )

    trainer = Trainer(model, cfg, train_loader, val_ds, device)
    if init_checkpoint is not None:
        extra = init_checkpoint.get("extra") or {}
        trainer.best_metric = float(init_checkpoint.get("best_metric", -1.0))
        trainer.best_r10 = float(extra.get("best_r10", -1.0))
        trainer.bad_evals = 0

        # Keep the warm-start checkpoint as the phase baseline. A phase that fails to
        # improve must still leave a valid best.pth instead of selecting a worse first eval.
        baseline_path = trainer.out_dir / "best.pth"
        source_path = Path(args.init_from)
        if source_path.resolve() != baseline_path.resolve():
            shutil.copy2(source_path, baseline_path)
        log.info(
            f"[init-from] baseline best mAP={trainer.best_metric:.4f} "
            f"best_R@10={trainer.best_r10:.4f} -> {baseline_path}"
        )
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
