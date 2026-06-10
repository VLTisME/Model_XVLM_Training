"""ITC — Image-Text Contrastive, a faithful port of X-VLM `get_contrastive_loss`.

analyze.md §4. Reference: X-VLM `models/xvlm.py::get_contrastive_loss`, ALBEF `model_pretrain.py`.

This is now exactly X-VLM's contrastive (review fix #4 removed the XBM memory bank):
  - learnable scalar temperature, used by DIVISION, `clamp_(0.001, 0.5)`. When wrapping real X-VLM
    we pass the backbone's PRETRAINED `temp` (external_temp), not a fresh one (review fix #6);
  - all_gather across GPUs (review fix #2): negatives = batch x world_size, via a
    gradient-preserving GatherLayer (no-op on a single process, so the math is unit-tested);
  - identity soft targets: every candidate sharing the anchor's id is a positive, normalized to
    sum 1 — X-VLM `pos_idx = eq(idx,idx.t()); labels = pos_idx/pos_idx.sum(1)`.

NOTE (single GPU): all_gather is a no-op, so the only negatives are the in-batch ones. Use a large
batch and/or multi-GPU for a strong contrastive (this is the X-VLM regime).
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


# ----------------------------------------------------------------- distributed gather
class _GatherLayer(torch.autograd.Function):
    """all_gather that preserves gradients to the local shard (MoCo/ALBEF pattern)."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return torch.cat(out, dim=0)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        grad = grad.contiguous()
        dist.all_reduce(grad)
        ws, rank = dist.get_world_size(), dist.get_rank()
        n = grad.shape[0] // ws
        return grad[rank * n:(rank + 1) * n]


def _dist_on() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _gather(x: Tensor) -> Tensor:
    return _GatherLayer.apply(x) if _dist_on() else x


def _gather_nograd(x: Tensor) -> Tensor:
    if not _dist_on():
        return x
    out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(out, x.contiguous())
    return torch.cat(out, dim=0)


class ITCLoss(nn.Module):
    def __init__(self, temp_init: float = 0.07, external_temp: Tensor | None = None):
        super().__init__()
        if external_temp is None:
            self.temp_param = nn.Parameter(torch.ones([]) * temp_init)
            self._ext: list[Tensor] | None = None
        else:
            # reference the backbone's pretrained temp WITHOUT re-registering it as our parameter
            self.temp_param = None
            self._ext = [external_temp]

    @property
    def temp(self) -> Tensor:
        return self._ext[0] if self._ext is not None else self.temp_param

    def forward(self, img_feat: Tensor, txt_feat: Tensor, ids: Tensor | None = None) -> Tensor:
        """
        Args:
            img_feat, txt_feat: [N, d] (re-normalized here to be safe).
            ids: [N] identity (sequence_id) per pair; default = globally-unique arange.
        """
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        device = img_feat.device
        n = img_feat.size(0)
        if ids is None:
            rank = dist.get_rank() if _dist_on() else 0
            ids = torch.arange(n, device=device) + rank * n   # globally unique default

        img = _gather(F.normalize(img_feat, dim=-1))
        txt = _gather(F.normalize(txt_feat, dim=-1))
        ids = _gather_nograd(ids)

        sim_i2t = img @ txt.t() / self.temp        # [Nall, Nall]
        sim_t2i = txt @ img.t() / self.temp

        pos = (ids[:, None] == ids[None, :]).float()
        targets = pos / pos.sum(dim=1, keepdim=True).clamp_min(1.0)

        loss_i2t = -(F.log_softmax(sim_i2t, dim=1) * targets).sum(dim=1).mean()
        loss_t2i = -(F.log_softmax(sim_t2i, dim=1) * targets).sum(dim=1).mean()
        return 0.5 * (loss_i2t + loss_t2i)

    @property
    def temperature(self) -> float:
        return float(self.temp.detach())
