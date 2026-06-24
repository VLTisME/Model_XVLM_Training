from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.data import BBoxAwareTransform, PairMixedBatchSampler  # noqa: E402
from star.metrics import full_report  # noqa: E402
from star.pe import (  # noqa: E402
    CrossBatchMemory,
    PEManifestDataset,
    PEPairCollator,
    PEVisionRetriever,
    pe_xbm_loss,
)
from star.utils import seed_everything  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train PE-Core-bigG-14-448 with XBM.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="hf-hub:timm/PE-Core-bigG-14-448")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--auto-batch", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--batch-candidates", default="4,6,8,10,12,16")
    parser.add_argument("--calibration-warmup", type=int, default=20)
    parser.add_argument("--calibration-steps", type=int, default=50)
    parser.add_argument("--target-effective-batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--xbm-size", type=int, default=4096)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--lora-start-epoch",
        type=int,
        default=1,
        help="enable visual LoRA from this zero-based epoch; use 0 for a warm-start phase",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--resume",
        default=None,
        help="exactly continue an interrupted run from last_pe.pth",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="load PE model weights only; use for a new recipe or additional epochs",
    )
    return parser.parse_args()


def loader_kwargs(args):
    kwargs = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def build_loader(dataset, batch_size, transform, args):
    pairs, groups = dataset.pairs()
    if not pairs:
        raise RuntimeError("PE training requires pair_image_id links in the train split")
    sampler = PairMixedBatchSampler(
        pairs,
        groups,
        batch_size=batch_size,
        hard_pairs=batch_size // 2,
        num_samples=len(dataset),
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=PEPairCollator(transform, hard_pairs=batch_size // 2),
        **loader_kwargs(args),
    )


def build_optimizer(model, args):
    head, lora = [], []
    for name, parameter in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora.append(parameter)
        elif parameter.requires_grad:
            head.append(parameter)
    return AdamW(
        [
            {"params": head, "lr": args.head_lr, "weight_decay": args.weight_decay},
            {"params": lora, "lr": args.lora_lr, "weight_decay": args.weight_decay},
        ],
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, total_steps, warmup_ratio):
    warmup = int(total_steps * warmup_ratio)

    def schedule(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, schedule)


def calibrate(model, dataset, transform, args, device):
    candidates = [int(value) for value in args.batch_candidates.split(",") if int(value) % 2 == 0]
    chosen = None
    results = []
    model.set_lora_enabled(True)
    for batch_size in candidates:
        loader = build_loader(dataset, batch_size, transform, args)
        iterator = iter(loader)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        try:
            for step in range(args.calibration_warmup + args.calibration_steps):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                image = batch["image"].to(device, non_blocking=True)
                text = F.normalize(batch["text_feature"].to(device), dim=-1)
                started = time.time()
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    image_feature = model.encode_image(image)
                    logits = image_feature @ text.t() / model.temperature
                    loss = 0.5 * (
                        F.cross_entropy(logits, torch.arange(batch_size, device=device))
                        + F.cross_entropy(logits.t(), torch.arange(batch_size, device=device))
                    )
                loss.backward()
                torch.cuda.synchronize()
                if step >= args.calibration_warmup:
                    elapsed.append(time.time() - started)
            peak = torch.cuda.max_memory_allocated() / 2**30
            speed = batch_size / (sum(elapsed) / len(elapsed))
            results.append({"batch_size": batch_size, "peak_gib": peak, "images_per_s": speed})
            print(f"calibration batch={batch_size}: peak={peak:.1f} GiB speed={speed:.1f} img/s")
            if peak <= 74:
                chosen = batch_size
        except torch.cuda.OutOfMemoryError:
            print(f"calibration batch={batch_size}: OOM")
            torch.cuda.empty_cache()
            break
        finally:
            del iterator
            del loader
            gc.collect()
    model.set_lora_enabled(False)
    if chosen is None:
        raise RuntimeError(f"No batch candidate fit. Results: {results}")
    print(f"selected physical batch: {chosen}")
    return chosen, results


@torch.no_grad()
def evaluate(model, dataset, transform, args, device):
    def collate(batch):
        return {
            "image": torch.stack(
                [transform.apply(item["image"], item["bbox"]) for item in batch]
            ),
            "text": torch.stack([item["text_feature"] for item in batch]),
            "image_id": [item["image_id"] for item in batch],
        }

    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate, **loader_kwargs(args))
    image_features, text_features, image_ids = [], [], []
    model.eval()
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image_features.append(model.encode_image(batch["image"].to(device)).float().cpu())
        text_features.append(F.normalize(batch["text"].float(), dim=-1))
        image_ids.extend(batch["image_id"])
    image_features = torch.cat(image_features)
    text_features = torch.cat(text_features)
    id_to_gallery, gallery_rows = {}, []
    for index, image_id in enumerate(image_ids):
        if image_id not in id_to_gallery:
            id_to_gallery[image_id] = len(gallery_rows)
            gallery_rows.append(index)
    frame = dataset.df
    query_rows = [index for index, value in enumerate(frame["caption"].fillna("")) if str(value).strip()]
    sim = text_features[query_rows] @ image_features[gallery_rows].t()
    gt = torch.tensor([id_to_gallery[image_ids[index]] for index in query_rows])
    return full_report(sim, gt, ks=(1, 5, 10, 50, 200))


def load_payload(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save(
    path,
    model,
    optimizer,
    scheduler,
    queue,
    epoch,
    step,
    best,
    args,
    report=None,
    checkpoint_kind="epoch_complete",
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "xbm": queue.state_dict(),
            "epoch": epoch,
            "step": step,
            "best": best,
            "args": vars(args),
            "report": report,
            "checkpoint_kind": checkpoint_kind,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
            },
        },
        path,
    )


def nvml_metrics(handle_info):
    if handle_info is None:
        return {}
    pynvml, handle = handle_info
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_utilization": float(utilization.gpu),
        "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        "gpu_temperature_c": float(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        ),
    }


def record_eval(
    model,
    val_set,
    eval_transform,
    args,
    device,
    log_path,
    tb,
    out_dir,
    optimizer,
    scheduler,
    queue,
    epoch,
    step,
    best,
    save_last_checkpoint,
):
    report = evaluate(model, val_set, eval_transform, args, device)
    print(f"[PE VAL-B] epoch={epoch} step={step} {report}")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "val", "epoch": epoch, "step": step, **report}) + "\n")
    if tb:
        for key, value in report.items():
            tb.add_scalar(f"val/{key}", value, step)
    if report["R@10"] > best["R@10"] or (
        report["R@10"] == best["R@10"] and report["R@1"] > best["R@1"]
    ):
        best = {"R@10": report["R@10"], "R@1": report["R@1"]}
        save(
            out_dir / "best_pe.pth",
            model,
            optimizer,
            scheduler,
            queue,
            epoch,
            step,
            best,
            args,
            report,
            checkpoint_kind="best_eval",
        )
        print(f"new best PE: {best}")
    if save_last_checkpoint:
        save(
            out_dir / "last_pe.pth",
            model,
            optimizer,
            scheduler,
            queue,
            epoch,
            step,
            best,
            args,
            report,
        )
    model.train()
    return best


def main():
    args = parse_args()
    if args.resume and args.init_from:
        raise SystemExit("Choose one of --resume or --init-from, not both.")

    resume_checkpoint = None
    if args.resume:
        if args.auto_batch or args.calibrate_only:
            raise SystemExit("--resume cannot be combined with --auto-batch or --calibrate-only.")
        resume_checkpoint = load_payload(args.resume, "cpu")
        kind = resume_checkpoint.get("checkpoint_kind")
        if kind == "best_eval" or (kind is None and Path(args.resume).name.startswith("best")):
            raise SystemExit(
                "--resume requires last_pe.pth from a completed epoch. "
                "Use --init-from for best_pe.pth."
            )
        saved_args = resume_checkpoint.get("args") or {}
        saved_model = saved_args.get("model")
        if saved_model and saved_model != args.model:
            print(f"[resume] restoring model={saved_model!r} from checkpoint")
            args.model = saved_model
        for name in (
            "batch_size",
            "target_effective_batch",
            "epochs",
            "xbm_size",
            "head_lr",
            "lora_lr",
            "weight_decay",
            "warmup_ratio",
            "lora_start_epoch",
            "workers",
            "prefetch_factor",
            "seed",
        ):
            if name in saved_args and getattr(args, name) != saved_args[name]:
                print(
                    f"[resume] restoring {name}={saved_args[name]!r} "
                    f"(CLI requested {getattr(args, name)!r})"
                )
                setattr(args, name, saved_args[name])
        for name in ("manifest", "image_root", "text_cache"):
            if name in saved_args and getattr(args, name) != saved_args[name]:
                print(
                    f"[resume] warning: {name} changed from {saved_args[name]!r} "
                    f"to {getattr(args, name)!r}; verify it points to the same data"
                )

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_metrics.jsonl"
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(str(out_dir / "tensorboard"))
    except ImportError:
        print("TensorBoard unavailable; continuing with JSONL logs.")
    nvml = None
    try:
        import pynvml

        pynvml.nvmlInit()
        nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device()))
    except Exception:
        print("NVML unavailable; install nvidia-ml-py for utilization/power/temperature logs.")

    train_set = PEManifestDataset(args.manifest, args.image_root, args.text_cache, split="train")
    val_set = PEManifestDataset(args.manifest, args.image_root, args.text_cache, split="valb")
    model = PEVisionRetriever(args.model).to(device)
    if args.init_from:
        checkpoint = load_payload(args.init_from, device)
        model.load_state_dict(checkpoint["model"], strict=True)
        print(
            f"[init-from] loaded PE weights from {args.init_from}; "
            "optimizer, scheduler, XBM, step, and RNG start fresh"
        )
    elif resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
    transform = BBoxAwareTransform(
        size=448, enabled=True, mean=model.image_mean, std=model.image_std
    )
    eval_transform = BBoxAwareTransform(
        size=448, enabled=False, mean=model.image_mean, std=model.image_std
    )
    print(model.trainable_summary())

    calibration = None
    if args.auto_batch or args.calibrate_only:
        args.batch_size, calibration = calibrate(model, train_set, transform, args, device)
        (out_dir / "batch_calibration.json").write_text(
            json.dumps(calibration, indent=2), encoding="utf-8"
        )
    if args.calibrate_only:
        return

    loader = build_loader(train_set, args.batch_size, transform, args)
    grad_accum = math.ceil(args.target_effective_batch / args.batch_size)
    steps_per_epoch = max(1, len(loader) // grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio)
    text_dim = train_set[0]["text_feature"].numel()
    queue = CrossBatchMemory(args.xbm_size, text_dim).to(device)
    start_epoch = step = 0
    best = {"R@10": -1.0, "R@1": -1.0}
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        queue.load_state_dict(resume_checkpoint["xbm"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        step = int(resume_checkpoint["step"])
        best = resume_checkpoint["best"]
        rng = resume_checkpoint.get("rng_state") or {}
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
        print(
            f"[resume] {args.resume}: completed epoch={start_epoch - 1}, "
            f"step={step}; continuing at epoch={start_epoch}/{args.epochs}"
        )

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        set_epoch = getattr(loader.batch_sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)
        model.set_lora_enabled(epoch >= args.lora_start_epoch)
        queue.reset()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data_end = time.time()
        optimizer_window_started = time.time()
        accumulated_data_time = 0.0
        midpoint = max(
            grad_accum,
            (max(1, steps_per_epoch * grad_accum // 2) // grad_accum) * grad_accum,
        )
        for iteration, batch in enumerate(loader):
            if iteration >= steps_per_epoch * grad_accum:
                break
            data_time = time.time() - data_end
            accumulated_data_time += data_time
            image = batch["image"].to(device, non_blocking=True)
            text = F.normalize(batch["text_feature"].to(device), dim=-1)
            instance = batch["instance_id"].to(device)
            hashes = batch["caption_hash"].to(device)
            partner = batch["partner_index"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                image_feature = model.encode_image(image)
                loss, diagnostics = pe_xbm_loss(
                    image_feature, text, instance, hashes, model.temperature, queue
                )
                hard_similarity = (image_feature * text[partner]).sum(dim=1).mean()
                scaled_loss = loss / grad_accum
            scaled_loss.backward()
            queue.enqueue(image_feature, text, instance, hashes)

            if (iteration + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    step_time = time.time() - optimizer_window_started
                    metrics = {
                        "kind": "train",
                        "epoch": epoch,
                        "step": step,
                        "loss": float(loss),
                        "positive_similarity": float(diagnostics["positive_similarity"]),
                        "paired_hard_similarity": float(hard_similarity),
                        "queue_size": int(queue.filled),
                        "queue_fill": float(diagnostics["queue_fill"]),
                        "queue_age": float(diagnostics["queue_age"]),
                        "queue_negative_similarity": float(
                            diagnostics["queue_negative_similarity"]
                        ),
                        "temperature": float(model.temperature),
                        "gradient_norm": float(grad_norm),
                        "lr_head": optimizer.param_groups[0]["lr"],
                        "lr_lora": optimizer.param_groups[1]["lr"],
                        "data_time_s": accumulated_data_time,
                        "step_time_s": step_time,
                        "images_per_s": args.batch_size * grad_accum / max(step_time, 1e-6),
                        "eta_minutes": (total_steps - step) * step_time / 60,
                        "vram_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                        "vram_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                        "vram_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                    metrics.update(nvml_metrics(nvml))
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(metrics, separators=(",", ":")) + "\n")
                    if tb:
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)) and key not in {"epoch", "step"}:
                                tb.add_scalar(f"train/{key}", value, step)
                    print(
                        f"e{epoch} s{step} loss={metrics['loss']:.3f} "
                        f"sim+={metrics['positive_similarity']:.3f} "
                        f"sim_h={metrics['paired_hard_similarity']:.3f} "
                        f"queue={metrics['queue_size']}/{args.xbm_size} "
                        f"img/s={metrics['images_per_s']:.1f} "
                        f"vram={metrics['vram_peak_gib']:.1f}G"
                    )
                optimizer_window_started = time.time()
                accumulated_data_time = 0.0
            if iteration + 1 == midpoint:
                best = record_eval(
                    model,
                    val_set,
                    eval_transform,
                    args,
                    device,
                    log_path,
                    tb,
                    out_dir,
                    optimizer,
                    scheduler,
                    queue,
                    epoch,
                    step,
                    best,
                    save_last_checkpoint=False,
                )
            data_end = time.time()

        best = record_eval(
            model,
            val_set,
            eval_transform,
            args,
            device,
            log_path,
            tb,
            out_dir,
            optimizer,
            scheduler,
            queue,
            epoch,
            step,
            best,
            save_last_checkpoint=True,
        )
    print(f"done in {(time.time() - started) / 3600:.2f}h; best={best}")
    if tb:
        tb.close()


if __name__ == "__main__":
    main()
