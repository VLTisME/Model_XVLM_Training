"""Trainer: AMP + grad-accum + grad-clip + warmup-cosine + VAL-B eval + early stop.

Includes a `--overfit-one-batch` sanity mode: a correct pipeline must drive the loss toward 0
on a single batch in a couple hundred steps; if it cannot, there is a wiring bug.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..utils.checkpoint import save_checkpoint
from ..utils.logging import get_logger
from .optim import build_optimizer, build_scheduler

log = get_logger("star.trainer")

_AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class Trainer:
    def __init__(self, model, cfg, train_loader: DataLoader, val_dataset=None, device="cuda"):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_dataset = val_dataset
        self.device = device

        self.steps_per_epoch = max(1, len(train_loader) // cfg.train.grad_accum)
        self.total_steps = self.steps_per_epoch * cfg.optim.epochs
        warmup_steps = int(self.steps_per_epoch * cfg.optim.warmup_epochs)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, self.total_steps, warmup_steps)

        self.amp_dtype = _AMP_DTYPE[cfg.train.amp_dtype]
        self.use_scaler = cfg.train.amp_dtype == "fp16"
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        if hasattr(torch.amp, "GradScaler"):                  # torch >= 2.4
            self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_scaler)
        else:                                                 # torch 2.1 (pinned X-VLM venv)
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_scaler)

        self.out_dir = Path(cfg.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -1.0
        self.best_r10 = -1.0
        self.bad_evals = 0
        self.step = 0
        self.start_epoch = 0
        self.max_seconds = None     # set via train.py --max-hours: graceful stop so a commit finishes < 9h
        self.stop_after_epoch = False
        self.nan_inf_count = 0
        self.metrics_path = self.out_dir / "train_metrics.jsonl"
        self.tb = None
        if cfg.train.log_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.out_dir / "tensorboard"))
            except ImportError:
                log.warning("TensorBoard is unavailable; continuing without it.")
        self.nvml = None
        if cfg.train.log_nvml and torch.cuda.is_available():
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml = (
                    pynvml,
                    pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device()),
                )
            except Exception:
                log.warning("NVML metrics unavailable; install nvidia-ml-py for GPU power/utilization.")

    # ------------------------------------------------------------------ helpers
    def resume_from(self, path: str) -> int:
        """Restore model+optimizer+scheduler+step+best from a checkpoint (last.pth) to continue a run
        ACROSS Kaggle commits. Returns the epoch to start from (derived from the saved step)."""
        from ..utils.checkpoint import load_checkpoint
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(path, map_location="cpu")
        extra = raw.get("extra") or {}
        saved_cfg = extra.get("cfg") or {}
        if saved_cfg:
            from ..config import to_dict

            current_cfg = to_dict(self.cfg)

            def read(config, dotted):
                value = config
                for part in dotted.split("."):
                    value = value[part]
                return value

            critical = (
                "data.image_size",
                "data.group_by",
                "data.pair_hard_pairs",
                "data.num_workers",
                "data.prefetch_factor",
                "data.persistent_workers",
                "model.backbone",
                "model.embed_dim",
                "model.lora_enabled",
                "model.lora_r",
                "model.lora_alpha",
                "model.lora_targets",
                "model.lora_freeze_text",
                "model.pose_enabled",
                "loss.w_itc",
                "loss.lambda_itm",
                "loss.lambda_smooth_ap",
                "loss.weighting",
                "optim.lr_lora",
                "optim.lr_head",
                "optim.weight_decay",
                "optim.warmup_epochs",
                "optim.epochs",
                "train.batch_size",
                "train.grad_accum",
                "train.amp_dtype",
                "train.seed",
            )
            mismatches = []
            for key in critical:
                try:
                    old, new = read(saved_cfg, key), read(current_cfg, key)
                except KeyError:
                    continue
                if old != new:
                    mismatches.append(f"{key}: checkpoint={old!r}, current={new!r}")
            if mismatches:
                details = "\n  ".join(mismatches)
                raise ValueError(
                    "Exact --resume requires the original training recipe. Differences:\n"
                    f"  {details}\nUse --init-from for a changed recipe."
                )
        kind = extra.get("checkpoint_kind")
        if kind == "best_eval" or (kind is None and Path(path).name.startswith("best")):
            raise ValueError(
                "--resume requires last.pth from a completed epoch. "
                "Use --init-from for best.pth or a changed recipe."
            )
        if kind is None and raw.get("step", 0) % self.steps_per_epoch:
            raise ValueError(
                "This legacy last.pth was saved in the middle of an epoch and cannot be "
                "resumed exactly. Use --init-from with this checkpoint, or resume from an "
                "epoch-boundary last.pth."
            )
        ckpt = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler, map_location=self.device
        )
        self.step = int(ckpt.get("step", 0))
        self.best_metric = float(ckpt.get("best_metric", -1.0))
        self.best_r10 = float(extra.get("best_r10", -1.0))
        self.bad_evals = int(extra.get("bad_evals", 0))
        self.nan_inf_count = int(extra.get("nan_inf_count", 0))
        saved_epoch = extra.get("epoch")
        self.start_epoch = int(saved_epoch) + 1 if saved_epoch is not None else self.step // self.steps_per_epoch
        log.info(f"[resume] {path}: step={self.step} best_mAP={self.best_metric:.4f} "
                 f"-> tiep tu epoch {self.start_epoch}/{self.cfg.optim.epochs}")
        return self.start_epoch
    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _forward_loss(self, batch: dict):
        with torch.autocast(device_type=self.device_type, dtype=self.amp_dtype,
                            enabled=self.amp_dtype != torch.float32):
            out = self.model(batch, step=self.step)
        return out

    # ------------------------------------------------------------------ sanity mode
    def overfit_one_batch(self, max_steps: int = 300, target: float = 0.05) -> float:
        batch = self._to_device(next(iter(self.train_loader)))
        self.model.train()
        # BUGFIX: the warmup scheduler initializes every group's LR to lr_lambda(0)=0, and this
        # loop never steps the scheduler -> the check would run at LR=0 and learn nothing.
        # A wiring check wants a constant healthy LR, independent of the schedule:
        for g in self.optimizer.param_groups:
            g["lr"] = 1e-3
        initial = None
        for i in range(max_steps):
            self.optimizer.zero_grad(set_to_none=True)
            out = self._forward_loss(batch)
            loss = out["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"NaN/Inf loss at step {i}: {out}")
            if initial is None:
                initial = loss.item()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if i % 25 == 0:
                vram = (f" vram={torch.cuda.max_memory_allocated()/2**30:.1f}G"
                        if torch.cuda.is_available() else "")
                log.info(f"[overfit] step {i:3d} loss={loss.item():.4f}{vram}")
            # success = absolute target OR a 70% relative drop. The absolute target is unreachable
            # when the batch holds same-instance duplicates: with k positives/row the ITC
            # soft-target loss has an irreducible floor of log(k) (e.g. log 2 = 0.693 with the
            # video-grouped sampler — observed exactly on the 10k_hard Kaggle run), so the
            # relative criterion must leave headroom above that floor.
            if loss.item() < target or loss.item() < 0.30 * initial:
                log.info(f"[overfit] OK: {initial:.3f} -> {loss.item():.4f} "
                         f"(target<{target} or 70% drop) at step {i}")
                return loss.item()
        log.warning(f"[overfit] did NOT converge ({initial:.3f} -> {loss.item():.4f}). NOTE: if the "
                    f"loss PLATEAUED near log(k) of same-id duplicates (e.g. ~0.69), wiring is fine; "
                    f"a flat or rising curve is the real red flag.")
        return loss.item()

    # ------------------------------------------------------------------ main loop
    def train(self):
        log.info(self.model.trainable_summary())
        eval_every = max(1, int(len(self.train_loader) * self.cfg.train.eval_every_epochs))
        accum = self.cfg.train.grad_accum
        t0 = time.time()
        last_batch_end = time.time()
        for epoch in range(self.start_epoch, self.cfg.optim.epochs):
            set_epoch = getattr(self.train_loader.batch_sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            optimizer_window_started = time.time()
            accumulated_data_time = 0.0
            for it, batch in enumerate(self.train_loader):
                if it >= self.steps_per_epoch * accum:
                    break
                data_time = time.time() - last_batch_end
                accumulated_data_time += data_time
                batch = self._to_device(batch)
                out = self._forward_loss(batch)
                loss = out["loss"] / accum
                if not torch.isfinite(loss):
                    log.error(f"NaN/Inf loss; skipping step. components={out}")
                    self.nan_inf_count += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # per-loss grad-norm diagnostic (train.grad_norm_every > 0): shows whether one
                # task dominates the gradient (the honest way to judge loss-weight balance)
                gne = self.cfg.train.grad_norm_every
                if gne and self.step % gne == 0 and (it % accum == 0):
                    self._log_grad_norms(out)

                self.scaler.scale(loss).backward()

                if (it + 1) % accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optim.grad_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.step += 1

                    if self.step % self.cfg.train.log_every_steps == 0:
                        metrics = self._step_metrics(
                            epoch,
                            out,
                            float(grad_norm),
                            accumulated_data_time,
                            time.time() - optimizer_window_started,
                            t0,
                        )
                        self._log_metrics(metrics)
                    optimizer_window_started = time.time()
                    accumulated_data_time = 0.0
                last_batch_end = time.time()

                if (it + 1) % eval_every == 0 and self.val_dataset is not None:
                    stop = self._evaluate_and_maybe_stop(epoch)
                    if self.max_seconds and (time.time() - t0) > self.max_seconds:
                        self.stop_after_epoch = True
                        log.info(
                            f"[time-stop] {(time.time()-t0)/3600:.2f}h vuot ngan sach; "
                            "se hoan tat epoch hien tai roi luu last.pth an toan."
                        )
                    if stop:
                        self.stop_after_epoch = True
                        log.info("Early-stop condition reached; finishing the current epoch.")
            self._save_last(epoch)
            if self.stop_after_epoch:
                log.info(f"Stopped safely after epoch {epoch}; resume with last.pth.")
                return
        log.info(f"Training done in {(time.time()-t0)/60:.1f} min. best mAP={self.best_metric:.4f}")
        if self.tb:
            self.tb.close()

    def _gpu_metrics(self) -> dict[str, float]:
        if not torch.cuda.is_available():
            return {}
        stats = {
            "vram_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "vram_reserved_gib": torch.cuda.memory_reserved() / 2**30,
            "vram_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        if self.nvml:
            pynvml, handle = self.nvml
            stats.update(
                {
                    "gpu_utilization": float(
                        pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    ),
                    "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                    "gpu_temperature_c": float(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                }
            )
        return stats

    def _step_metrics(self, epoch, out, grad_norm, data_time, step_time, started) -> dict:
        metrics = {
            "kind": "train",
            "epoch": epoch,
            "step": self.step,
            "lr": self.scheduler.get_last_lr()[0],
            "loss": float(out["loss"]),
            "itc": float(out["loss_itc"]),
            "itm": float(out["loss_itm"]),
            "smooth_ap": float(out["loss_smap"]),
            "positive_similarity": float(out.get("positive_similarity", 0)),
            "paired_hard_similarity": float(out.get("paired_hard_similarity", 0)),
            "random_negative_similarity": float(out.get("random_negative_similarity", 0)),
            "itm_positive_accuracy": float(out.get("itm_positive_accuracy", 0)),
            "itm_hard_image_accuracy": float(out.get("itm_hard_image_accuracy", 0)),
            "itm_hard_text_accuracy": float(out.get("itm_hard_text_accuracy", 0)),
            "temperature": float(out.get("temperature", 0)),
            "gradient_norm": grad_norm,
            "data_time_s": data_time,
            "step_time_s": step_time,
            "images_per_s": self.cfg.train.batch_size * self.cfg.train.grad_accum / max(step_time, 1e-6),
            "eta_minutes": max(0, self.total_steps - self.step) * step_time / 60,
            "elapsed_minutes": (time.time() - started) / 60,
            "nan_inf_count": self.nan_inf_count,
        }
        metrics.update(self._gpu_metrics())
        return metrics

    def _log_metrics(self, metrics: dict) -> None:
        log.info(
            " ".join(
                [
                    f"e{metrics['epoch']} s{metrics['step']}",
                    f"loss={metrics['loss']:.3f}",
                    f"itc={metrics['itc']:.3f}",
                    f"itm={metrics['itm']:.3f}",
                    f"smap={metrics['smooth_ap']:.3f}",
                    f"sim+={metrics['positive_similarity']:.3f}",
                    f"sim_h={metrics['paired_hard_similarity']:.3f}",
                    f"itm_acc={metrics['itm_positive_accuracy']:.2f}/"
                    f"{metrics['itm_hard_image_accuracy']:.2f}/"
                    f"{metrics['itm_hard_text_accuracy']:.2f}",
                    f"lr={metrics['lr']:.2e}",
                    f"img/s={metrics['images_per_s']:.1f}",
                    f"eta={metrics['eta_minutes']:.1f}m",
                    f"vram={metrics.get('vram_peak_gib', 0):.1f}G",
                    f"gpu={metrics.get('gpu_utilization', 0):.0f}%",
                ]
            )
        )
        if self.cfg.train.log_jsonl:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, separators=(",", ":")) + "\n")
        if self.tb:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in {"epoch", "step"}:
                    self.tb.add_scalar(f"train/{key}", value, self.step)

    def _save_last(self, epoch: int) -> None:
        from ..config import to_dict

        save_checkpoint(
            str(self.out_dir / "last.pth"),
            self.model,
            self.optimizer,
            self.scheduler,
            self.step,
            self.best_metric,
            {
                "cfg": to_dict(self.cfg),
                "epoch": epoch,
                "checkpoint_kind": "epoch_complete",
                "nan_inf_count": self.nan_inf_count,
                "best_r10": self.best_r10,
                "bad_evals": self.bad_evals,
            },
        )

    def _grad_norm_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _log_grad_norms(self, out: dict) -> None:
        """Per-loss gradient norms over trainable params (3 extra backwards; gated by config)."""
        params = self._grad_norm_params()
        norms = {}
        for key in ("loss_itc", "loss_itm", "loss_smap"):
            grads = torch.autograd.grad(out[key], params, retain_graph=True, allow_unused=True)
            sq = sum((g.float() ** 2).sum() for g in grads if g is not None)
            norms[key] = float(torch.sqrt(sq)) if isinstance(sq, torch.Tensor) else 0.0
        total = sum(norms.values()) or 1.0
        log.info("[grad-norm] " + " ".join(
            f"{k.removeprefix('loss_')}={v:.3e}({100 * v / total:.0f}%)" for k, v in norms.items()))

    def _evaluate_and_maybe_stop(self, epoch: int) -> bool:
        from .evaluator import evaluate_retrieval

        from ..config import to_dict

        rep = evaluate_retrieval(self.model, self.val_dataset, self.device,
                                 num_workers=self.cfg.data.num_workers)
        log.info(f"[VAL-B] {rep}")
        eval_metrics = {"kind": "val", "step": self.step, **rep}
        if self.cfg.train.log_jsonl:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_metrics, separators=(",", ":")) + "\n")
        if self.tb:
            for key, value in rep.items():
                self.tb.add_scalar(f"val/{key}", value, self.step)
        metric = rep["mAP"]
        previous_best_r10 = self.best_r10
        self.best_r10 = max(self.best_r10, rep["R@10"])
        r10_safe = (
            previous_best_r10 < 0
            or rep["R@10"] >= previous_best_r10 - self.cfg.train.best_r10_max_drop
        )
        # Embed the run config so evaluate.py can rebuild the exact architecture.
        # `last.pth` is intentionally written only after a complete epoch; a midpoint
        # DataLoader/augmentation state cannot be resumed exactly.
        cfg_dict = to_dict(self.cfg)
        if metric > self.best_metric and r10_safe:
            self.best_metric = metric
            self.bad_evals = 0
            save_checkpoint(str(self.out_dir / "best.pth"), self.model, self.optimizer,
                            self.scheduler, self.step, self.best_metric,
                            {"report": rep, "cfg": cfg_dict, "epoch": epoch,
                             "best_r10": self.best_r10,
                             "bad_evals": self.bad_evals,
                             "checkpoint_kind": "best_eval"})
            log.info(f"[VAL-B] new best mAP={metric:.4f} -> saved best.pth")
        else:
            self.bad_evals += 1
            if metric > self.best_metric and not r10_safe:
                log.info(
                    f"[VAL-B] mAP improved to {metric:.4f}, but R@10={rep['R@10']:.4f} "
                    f"violates safety floor {previous_best_r10 - self.cfg.train.best_r10_max_drop:.4f}"
                )
        self.model.train()
        return self.bad_evals >= self.cfg.train.early_stop_patience
