"""Hard-negative sampler safety (ALBEF-style)."""
import torch

from star.modules import sample_hard_negative


def test_hard_neg_excludes_forbidden():
    sim = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    forbid = torch.tensor([[True, False, False], [False, True, False]])  # forbid the diagonal
    idx = sample_hard_negative(sim, forbid, temperature=0.1)
    assert idx[0].item() != 0 and idx[1].item() != 1


def test_hard_neg_all_forbidden_fallback():
    sim = torch.randn(1, 3)
    forbid = torch.ones(1, 3, dtype=torch.bool)   # everything forbidden -> uniform fallback
    idx = sample_hard_negative(sim, forbid)
    assert 0 <= idx[0].item() < 3


def test_hard_neg_prefers_hardest_when_peaked():
    # at low temperature the hardest (highest-sim, non-forbidden) negative dominates
    sim = torch.tensor([[0.0, 9.0, 1.0]])         # col 1 is hardest
    forbid = torch.tensor([[True, False, False]])  # col 0 is the positive
    counts = torch.zeros(3)
    for _ in range(200):
        counts[sample_hard_negative(sim, forbid, temperature=0.05)[0].item()] += 1
    assert counts[1] > counts[2]                   # samples the hard negative far more often
