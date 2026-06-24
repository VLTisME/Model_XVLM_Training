from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import load_config, parse_overrides  # noqa: E402
from star.data import PABDataset, PairAwareCollator, PairMixedBatchSampler  # noqa: E402
from star.models import STARModel  # noqa: E402
from star.utils import seed_everything  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Probe safe physical X-VLM batch size on A100.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", default="32,48,64,80,96")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--target-vram-min", type=float, default=68)
    parser.add_argument("--target-vram-max", type=float, default=74)
    parser.add_argument("--output", default="batch_calibration_xvlm.json")
    parser.add_argument("--set", nargs="*", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    seed_everything(cfg.train.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    model = STARModel(cfg).to(device).train()
    tokenizer = model.backbone.tokenizer
    augmentation = {
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
    dataset = PABDataset(
        cfg.data.manifest,
        cfg.data.image_root,
        tokenizer,
        split="train",
        image_size=cfg.data.image_size,
        max_token=cfg.data.max_token,
        train=True,
        lhp_kwargs=augmentation,
        defer_transform=True,
    )
    pairs, groups = dataset.pairs()
    candidates = [int(value) for value in args.candidates.split(",")]
    results = []
    selected = None
    for batch_size in candidates:
        if batch_size % 2:
            continue
        sampler = PairMixedBatchSampler(
            pairs,
            groups,
            batch_size=batch_size,
            hard_pairs=batch_size // 2,
            num_samples=len(dataset),
            seed=cfg.train.seed,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=PairAwareCollator(dataset.transform, hard_pairs=batch_size // 2),
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=cfg.data.num_workers > 0,
            prefetch_factor=cfg.data.prefetch_factor if cfg.data.num_workers > 0 else None,
        )
        iterator = iter(loader)
        elapsed = []
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(args.warmup + args.steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                batch = {
                    key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                model.zero_grad(set_to_none=True)
                started = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(batch)["loss"]
                loss.backward()
                torch.cuda.synchronize()
                if step >= args.warmup:
                    elapsed.append(time.time() - started)
            peak = torch.cuda.max_memory_allocated() / 2**30
            speed = batch_size / (sum(elapsed) / len(elapsed))
            record = {"batch_size": batch_size, "peak_gib": peak, "images_per_s": speed}
            results.append(record)
            print(f"batch={batch_size}: peak={peak:.1f} GiB speed={speed:.1f} img/s")
            if peak <= args.target_vram_max:
                selected = batch_size
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": batch_size, "oom": True})
            print(f"batch={batch_size}: OOM")
            torch.cuda.empty_cache()
            break
        finally:
            del iterator
            del loader
            gc.collect()

    payload = {
        "selected_batch_size": selected,
        "target_vram_gib": [args.target_vram_min, args.target_vram_max],
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
