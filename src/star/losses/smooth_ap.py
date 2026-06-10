"""Smooth-AP — a differentiable approximation of Average Precision.

analyze.md §5. Paper: Brown et al. 2020 (2007.12163),
official code: github.com/Andrew-Brown1/Smooth_AP.

Idea: AP depends on integer ranks computed with a Heaviside step (non-differentiable).
Replace the step H(x) by a sigmoid sigma(x / tau):

    D[q,i,j]      = s_j - s_i
    R_all(q,i)    = 1 + sum_{j != i}             sigma(D[q,i,j] / tau)
    R_pos(q,i)    = 1 + sum_{j != i, rel[q,j]=1} sigma(D[q,i,j] / tau)
    AP_q          = (1/|P_q|) sum_{i: rel[q,i]=1} R_pos(q,i) / R_all(q,i)
    L_SmoothAP    = 1 - mean_q AP_q

Single-positive special case (this competition): R_pos == 1, so AP_q is the smooth
version of 1 / rank(GT) — exactly the mAP=MRR objective.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def smooth_ap_from_sim(sim: Tensor, relevance: Tensor, tau: float = 0.01) -> Tensor:
    """Compute the Smooth-AP loss from a similarity matrix.

    Args:
        sim:        [Q, M] query-to-gallery similarity (e.g., cosine).
        relevance:  [Q, M] in {0,1}; 1 where the gallery item is a positive.
        tau:        sigmoid temperature (smaller -> closer to the true step).
    Returns:
        scalar loss = 1 - mean AP (over queries that have >=1 positive).
    """
    q, m = sim.shape
    # D[q,i,j] = s_j - s_i   (unsqueeze(1)->vary j on last dim; unsqueeze(2)->vary i)
    d = sim.unsqueeze(1) - sim.unsqueeze(2)            # [Q, M(i), M(j)]
    sg = torch.sigmoid(d / tau)                        # [Q, M, M]

    off_diag = 1.0 - torch.eye(m, device=sim.device)   # zero out self (i==j)
    sg = sg * off_diag.unsqueeze(0)

    rank_all = 1.0 + sg.sum(dim=2)                      # [Q, M]
    rank_pos = 1.0 + (sg * relevance.unsqueeze(1)).sum(dim=2)

    ratio = rank_pos / rank_all.clamp_min(1e-6)         # [Q, M]
    num_pos = relevance.sum(dim=1)                      # [Q]
    ap = (ratio * relevance).sum(dim=1) / num_pos.clamp_min(1.0)

    valid = num_pos > 0
    if valid.sum() == 0:
        return sim.new_zeros(())
    return 1.0 - ap[valid].mean()


class SmoothAPLoss(nn.Module):
    """Convenience wrapper. Builds a text->image similarity within the batch and uses
    the (optionally duplicate-aware) identity as the relevance mask.
    """

    def __init__(self, tau: float = 0.01):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        img_feat: Tensor,
        txt_feat: Tensor,
        relevance: Tensor | None = None,
    ) -> Tensor:
        img = F.normalize(img_feat, dim=-1)
        txt = F.normalize(txt_feat, dim=-1)
        sim = txt @ img.t()                              # [N, N] text-query x image-gallery
        if relevance is None:
            relevance = torch.eye(sim.size(0), device=sim.device)
        return smooth_ap_from_sim(sim, relevance, self.tau)
