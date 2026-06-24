from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

from ..models.lora import LoRALinear


def load_pe_text_model(model_id: str):
    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit("PE requires: pip install open_clip_torch") from exc
    model, _, _ = open_clip.create_model_and_transforms(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    return model, tokenizer


def _block_index(name: str) -> int | None:
    match = re.search(r"(?:blocks|resblocks)\.(\d+)", name)
    return int(match.group(1)) if match else None


def _inject_upper_visual_lora(
    visual: nn.Module,
    upper_blocks: int,
    rank: int,
    alpha: int,
    dropout: float,
) -> int:
    indexed = [(name, _block_index(name)) for name, _ in visual.named_modules()]
    block_ids = sorted({index for _, index in indexed if index is not None})
    selected = set(block_ids[-upper_blocks:])
    count = 0
    for name, module in visual.named_modules():
        if _block_index(name) not in selected:
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in {"qkv", "query", "value"}:
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                count += 1
    return count


def _is_alignment_head(name: str) -> bool:
    name = name.lower()
    return (
        "attn_pool" in name
        or name.startswith("head")
        or name in {"proj", "projection"}
        or name.startswith("proj.")
        or name.startswith("projection.")
    )


class PEVisionRetriever(nn.Module):
    def __init__(
        self,
        model_id: str = "hf-hub:timm/PE-Core-bigG-14-448",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        upper_blocks: int = 8,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        model, _ = load_pe_text_model(model_id)
        self.model_id = model_id
        self.visual = model.visual
        preprocess_cfg = getattr(self.visual, "preprocess_cfg", {}) or {}
        self.image_mean = tuple(
            preprocess_cfg.get("mean", (0.48145466, 0.4578275, 0.40821073))
        )
        self.image_std = tuple(
            preprocess_cfg.get("std", (0.26862954, 0.26130258, 0.27577711))
        )
        self.logit_scale = nn.Parameter(model.logit_scale.detach().float().clone())
        del model

        for parameter in self.visual.parameters():
            parameter.requires_grad_(False)
        self.lora_layers = _inject_upper_visual_lora(
            self.visual, upper_blocks, lora_r, lora_alpha, lora_dropout
        )
        if self.lora_layers == 0:
            raise RuntimeError(
                "No upper-block attention projection was found for PE LoRA. "
                "Inspect the installed open_clip/timm model module names."
            )
        head_parameters = 0
        for name, parameter in self.visual.named_parameters():
            train_head = _is_alignment_head(name)
            train_lora = "lora_A" in name or "lora_B" in name
            parameter.requires_grad_(train_head or train_lora)
            if train_head and not train_lora:
                head_parameters += parameter.numel()
        if head_parameters == 0:
            raise RuntimeError("No PE attention-pool/projection/head parameters were selected.")
        self.set_lora_enabled(False)

        if grad_checkpointing:
            setter = getattr(self.visual, "set_grad_checkpointing", None)
            if callable(setter):
                setter(True)

    def set_lora_enabled(self, enabled: bool) -> None:
        for name, parameter in self.visual.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                parameter.requires_grad_(enabled)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        features = self.visual(image)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return F.normalize(features, dim=-1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """DataParallel-compatible image encoding entry point."""
        return self.encode_image(image)

    @property
    def temperature(self) -> torch.Tensor:
        return self.logit_scale.exp().reciprocal().clamp(1e-3, 0.5)

    def trainable_summary(self) -> str:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return (
            f"PE trainable={trainable / 1e6:.2f}M / total={total / 1e9:.2f}B "
            f"({100 * trainable / max(total, 1):.3f}%) lora_layers={self.lora_layers}"
        )
