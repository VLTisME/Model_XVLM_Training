"""Pairwise (duo) re-ranking comparator.

The cross-encoder scores each (query, image) POINTWISE -> when two images both match the query
its scores tie and rank-1 is decided by noise. This head compares TWO candidates head-to-head on
the cross-encoder's fused [CLS] features (`backbone.cross_feature`), so the top-K can be reordered
by a round-robin tournament (see star.inference.pairwise_rerank). The backbone stays frozen; only
this ~1M-param MLP trains, on the mined hard pairs (anchor's GT image vs its hard-negative image).

Antisymmetric by construction of the loss: head(a,b) trained to +, head(b,a) to - on (pos, neg).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class PairwiseHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, ha: Tensor, hb: Tensor) -> Tensor:
        """Logit of P(image a is a better match than image b | query). Inputs [..., dim]."""
        x = torch.cat([ha, hb, ha - hb], dim=-1)
        return self.net(x).squeeze(-1)
