"""Hard-negative sampling (ALBEF-style).

analyze.md §8. Sample a negative for each row from the similarity
distribution over candidates, *excluding* forbidden positions (the true match + duplicates).

Sampling (multinomial) rather than argmax keeps diversity and avoids over-fitting to a single
confusing negative; duplicate masking prevents treating a paraphrase of the positive as a negative.
"""
from __future__ import annotations

import torch
from torch import Tensor


def sample_hard_negative(sim: Tensor, forbid: Tensor, temperature: float = 0.5) -> Tensor:
    """For each row, sample one column index by softmax(sim/temperature), excluding `forbid`.

    Args:
        sim:    [N, M] similarity scores.
        forbid: [N, M] bool; True positions are never sampled (true match + dup captions).
        temperature: softmax temperature.
    Returns:
        [N] long tensor of sampled column indices.
    """
    logits = sim / temperature
    logits = logits.masked_fill(forbid, float("-inf"))
    probs = torch.softmax(logits, dim=1)

    # Review fix #10: small floor on non-forbidden entries (X-VLM adds 1e-5) so multinomial never
    # sees an all-zero row from softmax underflow on very peaked distributions.
    probs = probs + 1e-6 * (~forbid).to(probs.dtype)

    # Guard rows where everything is forbidden (e.g., tiny batch): fall back to uniform.
    bad = ~torch.isfinite(probs).all(dim=1) | (probs.sum(dim=1) == 0)
    if bad.any():
        valid = (~forbid).float()
        denom = valid.sum(dim=1, keepdim=True)
        fallback = torch.where(denom > 0, valid / denom.clamp_min(1.0),
                               torch.full_like(valid, 1.0 / sim.size(1)))
        probs = probs.clone()
        probs[bad] = fallback[bad]
    return torch.multinomial(probs, num_samples=1).squeeze(1)
