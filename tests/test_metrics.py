"""Metrics correctness — hand-computed values + cross-check vs sklearn."""
import torch
from sklearn.metrics import average_precision_score

from star.metrics import (full_report, mean_ap_multi, mean_ap_single,
                          rank_of_gt, recall_at_k)


def test_rank_of_gt_basic():
    sim = torch.tensor([[0.9, 0.1, 0.5],     # gt=0 -> rank 1
                        [0.2, 0.8, 0.5]])    # gt=2 -> rank 2
    ranks = rank_of_gt(sim, torch.tensor([0, 2]))
    assert ranks.tolist() == [1, 2]


def test_rank_pessimistic_on_ties():
    # all equal -> GT must get the worst-case rank (=3), not 1
    sim = torch.tensor([[0.5, 0.5, 0.5]])
    assert rank_of_gt(sim, torch.tensor([0])).item() == 3


def test_map_single_equals_mrr():
    sim = torch.tensor([[0.9, 0.1, 0.5], [0.2, 0.8, 0.5]])
    gt = torch.tensor([0, 2])
    # AP = mean(1/1, 1/2) = 0.75
    assert abs(mean_ap_single(sim, gt) - 0.75) < 1e-6


def test_recall_at_k():
    sim = torch.tensor([[0.9, 0.1, 0.5], [0.2, 0.8, 0.5]])
    gt = torch.tensor([0, 2])
    r = recall_at_k(sim, gt, ks=(1, 2))
    assert abs(r[1] - 0.5) < 1e-6 and abs(r[2] - 1.0) < 1e-6


def test_full_report_keys():
    sim = torch.rand(5, 20)
    rep = full_report(sim, torch.randint(0, 20, (5,)))
    assert {"mAP", "MRR", "R@1", "R@5", "R@10"} <= set(rep)


def test_map_multi_matches_sklearn():
    torch.manual_seed(0)
    q, g = 4, 15
    sim = torch.rand(q, g)
    rel = (torch.rand(q, g) > 0.7).float()
    rel[:, 0] = 1.0  # ensure each query has a positive
    ours = mean_ap_multi(sim, rel)
    ref = sum(
        average_precision_score(rel[i].numpy(), sim[i].numpy()) for i in range(q)
    ) / q
    assert abs(ours - ref) < 1e-5
