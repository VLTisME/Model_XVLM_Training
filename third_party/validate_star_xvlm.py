"""Validate the assembled STARModel on the REAL X-VLM backbone (run in the pinned venv).
.venv-xvlm/Scripts/python.exe third_party/validate_star_xvlm.py
"""
import sys

import torch

sys.path.insert(0, "src")
from star.config import load_config          # noqa: E402
from star.models import STARModel            # noqa: E402

cfg = load_config("configs/star_v3_100k.yaml")
cfg.model.backbone = "xvlm"
cfg.model.checkpoint = "data/checkpoints/xvlm_16m_base.th"

m = STARModel(cfg)
m.train()
print(m.trainable_summary())

frozen_keys = ("text_encoder.embeddings", "text_proj") + tuple(
    f"text_encoder.encoder.layer.{i}." for i in range(6))
text_frozen = all(not p.requires_grad for n, p in m.named_parameters()
                  if any(k in n for k in frozen_keys))
img_lora = any(p.requires_grad for n, p in m.named_parameters()
               if "vision_encoder" in n and ("lora_A" in n or "lora_B" in n))
cross_lora = any(p.requires_grad for n, p in m.named_parameters()
                 if "text_encoder.encoder.layer.6" in n and ("lora_A" in n or "lora_B" in n))
print("text tower frozen      :", text_frozen)
print("image (Swin) LoRA trains:", img_lora)
print("cross  (L6) LoRA trains :", cross_lora)
print("ITC uses backbone temp  :", m.itc._ext is not None)

tok = m.backbone.tokenizer
enc = tok(["a man is falling on the street", "a person running fast"],
          padding="max_length", truncation=True, max_length=cfg.data.max_token, return_tensors="pt")
batch = {"image": torch.randn(2, 3, cfg.data.image_size, cfg.data.image_size),
         "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
         "instance_id": torch.tensor([0, 1])}

out = m(batch, step=1)
print("losses:", {k: round(float(v), 4) for k, v in out.items()})
out["loss"].backward()
g = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None)
print("trainable params that received grad:", g)
assert text_frozen and img_lora and cross_lora and torch.isfinite(out["loss"])
print("XVLM STARMODEL VALIDATED")
