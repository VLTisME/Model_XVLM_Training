"""LoRA — low-rank adapters for nn.Linear (self-contained; `peft` optional).

analyze.md §3.   h = W0 x + (alpha/r) * B (A x)
A ~ N(0, .), B = 0  => starts as identity to the pretrained weights.
Only A, B are trainable; W0 frozen. Merge at inference: W <- W0 + (alpha/r) B A.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)            # delta = 0 at init
        self.merged = False

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        if self.merged:
            return out
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * delta

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.base.weight.data += self.scaling * (self.lora_B @ self.lora_A)
        self.merged = True


def inject_lora(
    model: nn.Module,
    targets: tuple[str, ...] = ("query", "value"),
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    exclude: tuple[str, ...] = (),
) -> int:
    """Replace nn.Linear submodules whose *name contains* any target substring with LoRALinear.

    Args:
        targets: substrings that select which Linear layers to adapt (e.g. attention 'query','value').
        exclude: substrings of the FULL module path to skip (e.g. 'text_encoder' to keep text frozen).
    Returns the number of layers wrapped.
    """
    count = 0
    for name, module in model.named_modules():
        if any(e in name for e in exclude):
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(t in child_name for t in targets):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                count += 1
    return count


def mark_only_lora_trainable(model: nn.Module, train_heads: tuple[str, ...] = ()) -> None:
    """Freeze everything except LoRA params and (optionally) named head submodules."""
    for n, p in model.named_parameters():
        is_lora = "lora_A" in n or "lora_B" in n
        is_head = any(h in n for h in train_heads)
        p.requires_grad_(is_lora or is_head)


@torch.no_grad()
def merge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def count_trainable(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
