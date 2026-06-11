"""Trainer: AMP + grad-accum + grad-clip + warmup-cosine + VAL-B eval + early stop.

Includes a `--overfit-one-batch` sanity mode: a correct pipeline must drive the loss toward 0
on a single batch in a couple hundred steps; if it cannot, there is a wiring bug.
"""
from __future__ import annotations

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

        steps_per_epoch = max(1, len(train_loader) // cfg.train.grad_accum)
        self.total_steps = steps_per_epoch * cfg.optim.epochs
        warmup_steps = int(steps_per_epoch * cfg.optim.warmup_epochs)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, self.total_steps, warmup_steps)

        self.amp_dtype = _AMP_DTYPE[cfg.train.amp_dtype]
        self.use_scaler = cfg.train.amp_dtype == "fp16"
        self.device_type = "cuda" if "cuda" in str(device) else "cpu"
        self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.use_scaler)

        self.out_dir = Path(cfg.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -1.0
        self.bad_evals = 0
        self.step = 0

    # ------------------------------------------------------------------ helpers
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
                log.info(f"[overfit] step {i:3d} loss={loss.item():.4f}")
            # success = absolute target OR a 75% relative drop. The absolute target is unreachable
            # when the batch holds same-instance duplicates: with k positives/row the ITC
            # soft-target loss has an irreducible floor of log(k), so we also accept the relative drop.
            if loss.item() < target or loss.item() < 0.25 * initial:
                log.info(f"[overfit] OK: {initial:.3f} -> {loss.item():.4f} "
                         f"(target<{target} or 75% drop) at step {i}")
                return loss.item()
        log.warning(f"[overfit] did NOT converge ({initial:.3f} -> {loss.item():.4f}); check wiring.")
        return loss.item()

    # ------------------------------------------------------------------ main loop
    def train(self):
        log.info(self.model.trainable_summary())
        eval_every = max(1, int(len(self.train_loader) * self.cfg.train.eval_every_epochs))
        accum = self.cfg.train.grad_accum
        t0 = time.time()
        for epoch in range(self.cfg.optim.epochs):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            for it, batch in enumerate(self.train_loader):
                batch = self._to_device(batch)
                out = self._forward_loss(batch)
                loss = out["loss"] / accum
                if not torch.isfinite(loss):
                    log.error(f"NaN/Inf loss; skipping step. components={out}")
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.step += 1

                    if self.step % 50 == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        msg = (f"e{epoch} s{self.step} loss={out['loss'].item():.3f} "
                               f"itc={out['loss_itc']:.3f} itm={out['loss_itm']:.3f} "
                               f"smap={out['loss_smap']:.3f} lr={lr:.2e}")
                        weights_fn = getattr(self.model.weighter, "weights", None)
                        if callable(weights_fn):          # dynamic weighting: show live weights
                            msg += f" w={weights_fn()}"
                        log.info(msg)

                if (it + 1) % eval_every == 0 and self.val_dataset is not None:
                    if self._evaluate_and_maybe_stop():
                        log.info("Early stopping.")
                        return
        log.info(f"Training done in {(time.time()-t0)/60:.1f} min. best mAP={self.best_metric:.4f}")

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

    def _evaluate_and_maybe_stop(self) -> bool:
        from .evaluator import evaluate_retrieval

        from ..config import to_dict

        rep = evaluate_retrieval(self.model, self.val_dataset, self.device,
                                 num_workers=self.cfg.data.num_workers)
        log.info(f"[VAL-B] {rep}")
        metric = rep["mAP"]
        # embed the run config so evaluate.py can rebuild the exact architecture (incl. overrides)
        cfg_dict = to_dict(self.cfg)
        save_checkpoint(str(self.out_dir / "last.pth"), self.model, self.optimizer,
                        self.scheduler, self.step, self.best_metric, {"cfg": cfg_dict})
        if metric > self.best_metric:
            self.best_metric = metric
            self.bad_evals = 0
            save_checkpoint(str(self.out_dir / "best.pth"), self.model, self.optimizer,
                            self.scheduler, self.step, self.best_metric,
                            {"report": rep, "cfg": cfg_dict})
            log.info(f"[VAL-B] new best mAP={metric:.4f} -> saved best.pth")
        else:
            self.bad_evals += 1
        self.model.train()
        return self.bad_evals >= self.cfg.train.early_stop_patience
