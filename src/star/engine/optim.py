"""Optimizer + warmup-cosine scheduler.

analyze.md §12. AdamW with param groups:
  - LoRA params:   lr_lora, weight decay
  - head params:   lr_head, weight decay
  - no-decay:      bias / LayerNorm / temperature / gate  (weight_decay = 0)
"""
from __future__ import annotations

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


_NO_DECAY_KEYS = ("bias", "norm", "layernorm", "temp", "gate", "img_pos", "txt_pos", "log_var")
# trainable task heads (text side is frozen, so no txt_proj; MLM removed)
_HEAD_KEYS = ("itm_head", "img_proj", "vision_proj", "pose", "weighter")


def _is_no_decay(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in _NO_DECAY_KEYS)


def _is_head(name: str) -> bool:
    return any(k in name for k in _HEAD_KEYS)


def build_optimizer(model, cfg) -> AdamW:
    groups = {
        "lora_decay": {"params": [], "lr": cfg.optim.lr_lora, "weight_decay": cfg.optim.weight_decay},
        "lora_nodecay": {"params": [], "lr": cfg.optim.lr_lora, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": cfg.optim.lr_head, "weight_decay": cfg.optim.weight_decay},
        "head_nodecay": {"params": [], "lr": cfg.optim.lr_head, "weight_decay": 0.0},
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_lora = "lora_A" in name or "lora_B" in name
        bucket = "lora" if is_lora else "head"   # non-LoRA trainable params = heads/branches
        suffix = "nodecay" if _is_no_decay(name) else "decay"
        groups[f"{bucket}_{suffix}"]["params"].append(p)
    param_groups = [g for g in groups.values() if g["params"]]
    return AdamW(param_groups, betas=tuple(cfg.optim.betas), eps=cfg.optim.eps)


def build_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    """Linear warmup then cosine decay to ~0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)
