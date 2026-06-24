"""Typed configuration: YAML -> dataclasses, with safe overrides.

Keeping config typed (instead of raw dicts) catches typos early and documents every knob.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    manifest: str = "manifests/star_v3.parquet"
    image_root: str = "data/pab"
    image_size: int = 384
    max_token: int = 100
    # LHP augmentation
    lhp_enabled: bool = False
    lhp_min_scale: float = 0.5
    lhp_use_bbox: bool = True
    pair_consistent_aug: bool = True
    motion_blur_p: float = 0.25
    jpeg_p: float = 0.35
    downscale_p: float = 0.30
    color_jitter_p: float = 0.30
    noise_p: float = 0.15
    erase_p: float = 0.20
    max_aug_ops: int = 2
    # smart sampler
    group_by: str = "scene"          # "scene" | "action" | "none" | "pair" | "pair_mixed"
    group_fraction: float = 0.5       # fraction of batch drawn from one group
    pair_hard_pairs: int = -1         # -1 = batch_size//2, every row has one hard partner
    num_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True


@dataclass
class ModelConfig:
    backbone: str = "xvlm"            # "xvlm" | "dummy"
    checkpoint: str | None = None     # path to X-VLM / CMP weights
    xvlm_repo: str | None = None      # external zengyan-97/X-VLM source checkout
    embed_dim: int = 256
    # LoRA
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("query", "value")
    lora_freeze_text: bool = True     # PLAN: text encoder FROZEN (no LoRA) — adapt image+cross only
    # pose branch (toggle) — fused into the IMAGE-encoder branch, no separate pose loss
    pose_enabled: bool = False
    pose_hidden: int = 256


@dataclass
class LossConfig:
    # PLAN total loss:  L = ITC + lambda_itm * ITM + lambda_smooth_ap * Smooth-AP   (MLM removed)
    w_itc: float = 1.0                # ITC coefficient (fixed at 1.0 in the plan)
    lambda_itm: float = 1.0           # λ1 — ITC:ITM = 1:1 is the proven X-VLM/ALBEF ratio: keep
    lambda_smooth_ap: float = 0.3     # λ2 — the ONE real unknown: sweep {0, 0.1, 0.3, 1.0} on VAL-B
    # weighting scheme: "fixed" (default, recommended) | "uncertainty" (Kendall 1705.07115)
    # | "dwa" (Liu 1803.10704). Dynamic modes apply ON TOP of the base weights above (ablation).
    weighting: str = "fixed"
    dwa_temp: float = 2.0
    itc_temp_init: float = 0.07
    smooth_ap_temp: float = 0.01


@dataclass
class OptimConfig:
    lr_lora: float = 2e-4
    lr_head: float = 4e-4
    weight_decay: float = 0.02
    betas: tuple[float, float] = (0.9, 0.999)   # matches X-VLM/ALBEF AdamW (unmodified default)
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_epochs: float = 1.0
    epochs: int = 8


@dataclass
class TrainConfig:
    batch_size: int = 24
    grad_accum: int = 4
    amp_dtype: str = "bf16"           # "bf16" | "fp16" | "fp32"
    grad_checkpointing: bool = True
    eval_every_epochs: float = 0.5
    early_stop_patience: int = 2
    grad_norm_every: int = 0          # >0: every N optimizer steps, log per-loss grad norms
    seed: int = 42
    out_dir: str = "outputs/star_v3"
    log_wandb: bool = False
    log_every_steps: int = 25
    log_jsonl: bool = True
    log_tensorboard: bool = True
    log_nvml: bool = True
    best_r10_max_drop: float = 0.002


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ----------------------------------------------------------------------------- helpers
def _merge(dc: Any, overrides: dict[str, Any]) -> Any:
    """Recursively apply a (possibly nested) dict onto a dataclass instance."""
    if not dataclasses.is_dataclass(dc):
        return overrides
    for k, v in overrides.items():
        if not hasattr(dc, k):
            raise KeyError(f"Unknown config key '{k}' for {type(dc).__name__}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(dc, k, tuple(v))
        else:
            setattr(dc, k, v)
    return dc


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML file into a typed Config; optional CLI overrides on top."""
    cfg = Config()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    _merge(cfg, raw)
    if overrides:
        _merge(cfg, overrides)
    return cfg


def to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse CLI `--set a.b=1 c=foo` pairs into a nested dict (shared by train/evaluate)."""
    import ast

    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, val = pair.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            # ast.literal_eval doesn't know YAML/shell-style lowercase bools/null -> handle them,
            # else `--set flag=false` becomes the STRING "false" (which is TRUTHY -> flag stays on).
            low = val.strip().lower()
            if low in ("true", "false"):
                val = low == "true"
            elif low in ("null", "none", "~"):
                val = None
            # otherwise keep as a plain string
        node[parts[-1]] = val
    return out
