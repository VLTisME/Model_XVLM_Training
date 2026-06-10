"""Retrieval metrics: mAP / MRR / R@K.

Competition metric = mAP. With exactly 1 ground-truth image per query, mAP == MRR
(see analyze.md §1). We implement:
  - rank_of_gt:      1-based rank of the GT for each query (worst-case on ties)
  - recall_at_k:     fraction of queries with GT in top-K
  - mean_ap_single:  mAP for the single-positive case (== MRR)
  - mean_ap_multi:   general mAP with a relevance mask (cross-checked vs sklearn in tests)

All functions take a similarity matrix S of shape [Q, G] (higher = more similar).
"""
from __future__ import annotations

import torch
from torch import Tensor


def rank_of_gt(sim: Tensor, gt_index: Tensor) -> Tensor:
    """1-based rank of the ground-truth gallery item for each query.

    Ties are broken pessimistically: rank = 1 + (#items strictly greater) + (#ties).
    This avoids optimistic scoring when many gallery items share a score.

    Args:
        sim:      [Q, G] similarity (query x gallery).
        gt_index: [Q] index into gallery of the correct item.
    Returns:
        ranks: [Q] long tensor, 1-based.
    """
    q = torch.arange(sim.size(0), device=sim.device)
    gt_score = sim[q, gt_index].unsqueeze(1)             # [Q,1]
    greater = (sim > gt_score).sum(dim=1)                # strictly better items
    ties = (sim == gt_score).sum(dim=1) - 1              # equal items excluding GT itself
    return greater + ties + 1                            # pessimistic 1-based rank


def recall_at_k(sim: Tensor, gt_index: Tensor, ks=(1, 5, 10)) -> dict[int, float]:
    ranks = rank_of_gt(sim, gt_index)
    return {k: (ranks <= k).float().mean().item() for k in ks}


def mean_ap_single(sim: Tensor, gt_index: Tensor) -> float:
    """mAP for single-positive == MRR == mean(1 / rank(GT))."""
    ranks = rank_of_gt(sim, gt_index).float()
    return (1.0 / ranks).mean().item()


def mrr(sim: Tensor, gt_index: Tensor) -> float:
    return mean_ap_single(sim, gt_index)


def mean_ap_multi(sim: Tensor, relevance: Tensor) -> float:
    """General mAP averaged over queries, with a binary relevance matrix.

    Args:
        sim:        [Q, G]
        relevance:  [Q, G] in {0,1}; 1 = positive for that query.
    Returns:
        mAP (float). Queries with no positive are skipped.
    """
    q_aps = []
    order = torch.argsort(sim, dim=1, descending=True)            # [Q, G]
    rel_sorted = torch.gather(relevance, 1, order).float()        # [Q, G]
    csum = torch.cumsum(rel_sorted, dim=1)                        # positives seen so far
    ranks = torch.arange(1, sim.size(1) + 1, device=sim.device).float()
    precision_at_hits = csum / ranks                              # Prec@k for every k
    num_pos = rel_sorted.sum(dim=1)                               # [Q]
    for i in range(sim.size(0)):
        if num_pos[i] == 0:
            continue
        ap = (precision_at_hits[i] * rel_sorted[i]).sum() / num_pos[i]
        q_aps.append(ap)
    if not q_aps:
        return 0.0
    return torch.stack(q_aps).mean().item()


def full_report(sim: Tensor, gt_index: Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    out = {"mAP": mean_ap_single(sim, gt_index), "MRR": mrr(sim, gt_index)}
    out.update({f"R@{k}": v for k, v in recall_at_k(sim, gt_index, ks).items()})
    return out
