"""Loss-weighting schemes: fixed / uncertainty (Kendall) / DWA (Liu)."""
import torch

from star.losses import DWAWeighter, FixedWeighter, UncertaintyWeighter, build_weighter

BASE = {"itc": 1.0, "itm": 1.0, "smap": 0.3}


def _losses(itc=2.0, itm=0.7, smap=0.4):
    return {k: torch.tensor(v, requires_grad=True)
            for k, v in zip(("itc", "itm", "smap"), (itc, itm, smap))}


def test_fixed_matches_manual():
    total = FixedWeighter(BASE)(_losses())
    assert torch.isclose(total, torch.tensor(1.0 * 2.0 + 1.0 * 0.7 + 0.3 * 0.4))


def test_uncertainty_equals_fixed_at_init_and_gets_grad():
    w = UncertaintyWeighter(BASE)
    total = w(_losses())
    # s_i = 0 => exp(-0)=1 and the +s_i regularizer is 0 => exactly the fixed loss
    assert torch.isclose(total, torch.tensor(2.0 + 0.7 + 0.12))
    total.backward()
    assert w.log_var.grad is not None and torch.isfinite(w.log_var.grad).all()
    assert set(w.weights()) == {"itc", "itm", "smap"}


def test_dwa_equal_weights_until_history_then_upweights_slow_task():
    w = DWAWeighter(BASE, temp=2.0)
    w(_losses(2.0, 0.7, 0.4))
    w(_losses(1.0, 0.7, 0.4))           # itc descends fast; itm/smap flat
    w(_losses(0.9, 0.7, 0.4))           # weights from history now apply
    k = w.weights()
    assert abs(sum(k.values()) - 3.0) < 1e-4          # softmax * K sums to K
    assert k["itc"] < k["itm"]                        # fast-descending itc down-weighted


def test_build_weighter_dispatch_and_unknown():
    class L:  # minimal cfg.loss stand-in
        w_itc, lambda_itm, lambda_smooth_ap, dwa_temp = 1.0, 1.0, 0.3, 2.0
        weighting = "uncertainty"
    assert isinstance(build_weighter(L), UncertaintyWeighter)
    L.weighting = "dwa"
    assert isinstance(build_weighter(L), DWAWeighter)
    L.weighting = "bogus"
    try:
        build_weighter(L)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_model_with_uncertainty_weighting_trains():
    from star.config import Config
    from star.models import STARModel
    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.embed_dim = 64
    cfg.loss.weighting = "uncertainty"
    m = STARModel(cfg)
    batch = {
        "image": torch.randn(3, 3, 384, 384),
        "input_ids": torch.randint(5, 900, (3, 16)),
        "attention_mask": torch.ones(3, 16, dtype=torch.long),
        "instance_id": torch.arange(3),
    }
    out = m(batch, step=1)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert m.weighter.log_var.grad is not None        # the weighter actually learns
