"""Multi-task loss weighting: fixed / uncertainty (Kendall) / DWA (Liu).

analyze.md §14. The PROVEN default is fixed weights (X-VLM/ALBEF use ITC:ITM = 1:1). The dynamic
schemes are ABLATION options and are applied ON TOP of the fixed base weights, so the proven 1:1
prior is kept and the scheme only adapts around it:

  fixed:        L = Σ w_i · L_i
  uncertainty:  L = Σ exp(-s_i) · (w_i · L_i) + s_i        (Kendall, arXiv 1705.07115)
                s_i = log σ_i², learnable, init 0 ⇒ EXACTLY the fixed loss at start.
                Equilibrium pushes each weighted loss toward ~1 (known caveat: can
                down-weight the hardest task — that is why it stays an ablation).
  dwa:          L = Σ k_i · (w_i · L_i),  k_i = K · softmax(r_i / T),
                r_i = L_i(t-1) / L_i(t-2)                   (Liu, arXiv 1803.10704)
                Slowly-descending losses get up-weighted. Stateful but not learnable;
                the 2-step history is NOT checkpointed (resume restarts at equal weights).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

TASKS = ("itc", "itm", "smap")


class FixedWeighter(nn.Module):
    def __init__(self, base: dict[str, float]):
        super().__init__()
        self.base = dict(base)

    def forward(self, losses: dict[str, Tensor]) -> Tensor:
        return sum(self.base[k] * losses[k] for k in TASKS)


class UncertaintyWeighter(nn.Module):
    """Kendall et al. 2018 — learn s_i = log σ_i²; precision exp(-s_i) reweights each task."""

    def __init__(self, base: dict[str, float]):
        super().__init__()
        self.base = dict(base)
        self.log_var = nn.Parameter(torch.zeros(len(TASKS)))

    def forward(self, losses: dict[str, Tensor]) -> Tensor:
        total = losses[TASKS[0]].new_zeros(())
        for i, k in enumerate(TASKS):
            total = total + torch.exp(-self.log_var[i]) * self.base[k] * losses[k] + self.log_var[i]
        return total

    def weights(self) -> dict[str, float]:
        with torch.no_grad():
            return {k: round(float(torch.exp(-self.log_var[i])), 4) for i, k in enumerate(TASKS)}


class DWAWeighter(nn.Module):
    """Liu et al. 2019 (DWA) — weight ∝ how slowly a loss has been descending recently."""

    def __init__(self, base: dict[str, float], temp: float = 2.0):
        super().__init__()
        self.base = dict(base)
        self.temp = temp
        self._hist: list[dict[str, float]] = []     # last two loss values per task
        self.last_k: dict[str, float] = {k: 1.0 for k in TASKS}

    def forward(self, losses: dict[str, Tensor]) -> Tensor:
        if len(self._hist) >= 2:
            prev1, prev2 = self._hist[-1], self._hist[-2]
            r = torch.tensor([prev1[k] / max(prev2[k], 1e-8) for k in TASKS])
            k_w = len(TASKS) * torch.softmax(r / self.temp, dim=0)
        else:
            k_w = torch.ones(len(TASKS))            # equal weights until history exists
        self.last_k = {k: round(float(k_w[i]), 4) for i, k in enumerate(TASKS)}

        total = losses[TASKS[0]].new_zeros(())
        for i, k in enumerate(TASKS):
            total = total + float(k_w[i]) * self.base[k] * losses[k]
        self._hist.append({k: float(losses[k].detach()) for k in TASKS})
        self._hist = self._hist[-2:]
        return total

    def weights(self) -> dict[str, float]:
        return dict(self.last_k)


def build_weighter(cfg_loss) -> nn.Module:
    base = {"itc": cfg_loss.w_itc, "itm": cfg_loss.lambda_itm, "smap": cfg_loss.lambda_smooth_ap}
    mode = cfg_loss.weighting
    if mode == "fixed":
        return FixedWeighter(base)
    if mode == "uncertainty":
        return UncertaintyWeighter(base)
    if mode == "dwa":
        return DWAWeighter(base, cfg_loss.dwa_temp)
    raise ValueError(f"unknown loss.weighting='{mode}' (expected fixed|uncertainty|dwa)")
