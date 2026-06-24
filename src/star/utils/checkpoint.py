"""Checkpoint save/load with run metadata."""
from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


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
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state() if np is not None else None,
        },
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
    restore_rng: bool = True,
) -> dict:
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if restore_rng:
        rng = ckpt.get("rng_state") or {}
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        if np is not None and rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
    return ckpt
