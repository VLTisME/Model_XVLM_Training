"""Checkpoint save/load with run metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str, model, optimizer=None, scheduler=None, step: int = 0,
                    best_metric: float = 0.0, extra: dict[str, Any] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "best_metric": best_metric,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
