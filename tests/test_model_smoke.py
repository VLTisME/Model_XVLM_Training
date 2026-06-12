"""End-to-end wiring test on the dummy backbone (no downloads, CPU-friendly).

Proves, for the PLAN architecture (L = ITC + λ1·ITM + λ2·SmoothAP, MLM removed, text frozen):
  - forward returns finite losses for the 3 heads
  - the text tower is frozen (requires_grad == False)
  - a few optimizer steps on a fixed batch reduce the loss (the pipeline can overfit one batch)
"""
import torch

from star.config import Config
from star.models import STARModel


def _tiny_cfg() -> Config:
    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.checkpoint = None
    cfg.model.embed_dim = 64
    cfg.model.lora_enabled = True          # dummy has no Q/V Linear -> n_lora=0, trains fully...
    cfg.model.lora_freeze_text = True      # ...except the text tower, which must be frozen
    return cfg


def _batch(b=4, L=16):
    return {
        "image": torch.randn(b, 3, 384, 384),
        "input_ids": torch.randint(5, 900, (b, L)),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "instance_id": torch.arange(b),
    }


def test_forward_returns_plan_losses():
    model = STARModel(_tiny_cfg())
    out = model(_batch(), step=1)
    assert set(out) == {"loss", "loss_itc", "loss_itm", "loss_smap"}   # no loss_mlm
    for v in out.values():
        assert torch.isfinite(torch.as_tensor(v))


def test_text_tower_is_frozen():
    model = STARModel(_tiny_cfg())
    text_keys = ("tok_embed", "txt_pos", "text_self", "txt_proj")
    frozen = [n for n, p in model.named_parameters()
              if any(k in n for k in text_keys)]
    assert frozen, "expected to find text-tower params"
    assert all(not p.requires_grad for n, p in model.named_parameters()
               if any(k in n for k in text_keys)), "text tower must be frozen"
    # image side must still be trainable
    assert any(p.requires_grad for n, p in model.named_parameters() if "patch" in n or "img_proj" in n)


def test_pose_fused_at_eval_when_enabled():
    # train/eval consistency: with the pose branch on, encode_for_eval MUST fuse keypoints
    cfg = _tiny_cfg()
    cfg.model.pose_enabled = True
    model = STARModel(cfg)
    model.eval()
    img = torch.randn(2, 3, 384, 384)
    ids = torch.randint(5, 900, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    kpts = torch.rand(2, 51)
    with torch.no_grad():
        f_plain, _ = model.encode_for_eval(img, ids, mask)
        f_pose, _ = model.encode_for_eval(img, ids, mask, keypoints=kpts)
    assert not torch.allclose(f_plain, f_pose), "pose branch must change eval image features"


def test_overfit_one_batch_decreases_loss():
    torch.manual_seed(0)
    model = STARModel(_tiny_cfg())
    batch = _batch()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    first = model(batch, step=1)["loss"].item()
    for step in range(60):
        opt.zero_grad()
        loss = model(batch, step=step + 1)["loss"]
        loss.backward()
        opt.step()
    last = model(batch, step=100)["loss"].item()
    assert last < first * 0.7, f"loss did not drop enough: {first:.3f} -> {last:.3f}"
