"""ITM — Image-Text Matching with hard negatives.

analyze.md §6. Paper: ALBEF (2107.07651).

The cross-encoder fuses each (image, text) pair into a [CLS] vector; a 2-way head predicts
match / no-match. We construct, per batch item, one hard-negative image and one hard-negative
text (sampled from the ITC similarity distribution -> "the pairs the bi-encoder confuses most").

This module is split so it is testable without the cross-encoder:
  - `build_itm_pairs(sim_i2t, ...)`  -> indices + labels for positive/negative pairs
  - `ITMLoss`                        -> 2-way cross-entropy on the head logits
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..modules.hard_neg import sample_hard_negative


def build_itm_pairs(
    sim_i2t: Tensor,
    dup_mask: Tensor | None = None,
    temperature: float = 1.0,
) -> dict[str, Tensor]:
    """Build (image_idx, text_idx, label) triplets for ITM.

    For a batch of N matched pairs we produce 3N pairs:
      - N positives:           (i, i, 1)
      - N hard-neg texts:      (i, t_neg(i), 0)   t_neg sampled by sim_i2t[i]
      - N hard-neg images:     (i_neg(t), t, 0)   symmetric, via sim_i2t.T

    Args:
        sim_i2t:  [N, N] image->text similarity. To match X-VLM, pass it ALREADY divided by the
                  model temperature (then keep temperature=1.0) so sampling is peaked on the
                  hardest negatives (review fix #1).
        dup_mask: [N, N] bool, True where text j is a duplicate caption of pair i
                  (those must NOT be used as negatives). Optional.
    Returns:
        dict with 'img_idx', 'txt_idx', 'label' (all length 3N).
    """
    n = sim_i2t.size(0)
    device = sim_i2t.device
    diag = torch.arange(n, device=device)

    # forbid the true match (and duplicates) from being chosen as a negative
    forbid = torch.eye(n, dtype=torch.bool, device=device)
    if dup_mask is not None:
        forbid = forbid | dup_mask

    neg_txt = sample_hard_negative(sim_i2t, forbid, temperature)        # [N] for each image
    neg_img = sample_hard_negative(sim_i2t.t(), forbid.t(), temperature)  # [N] for each text

    img_idx = torch.cat([diag, diag, neg_img])
    txt_idx = torch.cat([diag, neg_txt, diag])
    label = torch.cat([
        torch.ones(n, device=device),
        torch.zeros(n, device=device),
        torch.zeros(n, device=device),
    ]).long()
    return {"img_idx": img_idx, "txt_idx": txt_idx, "label": label}


def build_explicit_itm_pairs(partner_index: Tensor) -> dict[str, Tensor]:
    """Build 3N examples from one explicit in-batch hard partner per row."""
    if partner_index.ndim != 1:
        raise ValueError("partner_index must be a 1D tensor")
    n = partner_index.numel()
    device = partner_index.device
    diag = torch.arange(n, device=device)
    if n == 0 or (partner_index < 0).any() or (partner_index >= n).any():
        raise ValueError("every row must have one valid in-batch partner")
    if (partner_index == diag).any():
        raise ValueError("a row cannot be its own hard partner")
    return {
        "img_idx": torch.cat([diag, diag, partner_index]),
        "txt_idx": torch.cat([diag, partner_index, diag]),
        "label": torch.cat(
            [
                torch.ones(n, device=device),
                torch.zeros(n, device=device),
                torch.zeros(n, device=device),
            ]
        ).long(),
    }


class ITMLoss(nn.Module):
    """2-way cross-entropy on cross-encoder match logits."""

    def __init__(self):
        super().__init__()

    def forward(self, match_logits: Tensor, labels: Tensor) -> Tensor:
        """
        Args:
            match_logits: [P, 2] logits from the ITM head over the P built pairs.
            labels:       [P] in {0,1}.
        """
        return F.cross_entropy(match_logits, labels)
