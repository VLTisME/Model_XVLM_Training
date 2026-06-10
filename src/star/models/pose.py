"""Pose branch (toggle).

analyze.md §11. 17 keypoints (x,y,conf) -> MLP -> f_pose, gated-fused into f_V:
    f_V' = LayerNorm(f_V + g ⊙ W_p f_pose),  g in (0,1) learnable gate.
Kept as an optional branch; gain on PAB is unproven (verify on VAL-B).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class PoseBranch(nn.Module):
    def __init__(self, embed_dim: int, n_keypoints: int = 17, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_keypoints * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Parameter(torch.zeros(1))   # sigmoid(0)=0.5; starts modest
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_feat: Tensor, keypoints: Tensor) -> Tensor:
        """
        Args:
            img_feat:  [B, d]
            keypoints: [B, 17*3] normalized (x, y, conf)
        Returns:
            fused [B, d]
        """
        f_pose = self.proj(self.encoder(keypoints))
        g = torch.sigmoid(self.gate)
        return self.norm(img_feat + g * f_pose)
