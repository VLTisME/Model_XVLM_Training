"""Feasibility probe: build X-VLM retrieval model, load the 16M checkpoint, run a forward.
Run with the pinned venv:  .venv-xvlm/Scripts/python.exe third_party/probe_xvlm.py
"""
import os
import sys

import torch

REPO = os.path.join(os.path.dirname(__file__), "X-VLM")
CKPT = os.path.join(os.path.dirname(__file__), "..", "data", "checkpoints", "xvlm_16m_base.th")
sys.path.insert(0, REPO)

print("torch", torch.__version__)
import transformers  # noqa: E402
print("transformers", transformers.__version__)

from models.model_retrieval import XVLM  # noqa: E402

config = {
    "use_swin": True, "use_clip_vit": False, "use_roberta": False,
    "image_res": 384, "patch_size": 32,
    "vision_config": os.path.join(REPO, "configs", "config_swinB_384.json"),
    "text_config": os.path.join(REPO, "configs", "config_bert.json"),
    "text_encoder": "bert-base-uncased",
    "embed_dim": 256, "temp": 0.07,
}

print("building XVLM ...")
model = XVLM(config=config)
print("loading checkpoint ...")
model.load_pretrained(CKPT, config, is_eval=False)
model.eval()

# count modules for LoRA scoping sanity
n_qkv = sum(1 for n, _ in model.named_modules() if n.endswith("attn.qkv"))
n_cross_qv = sum(1 for n, _ in model.named_modules()
                 if ("text_encoder.encoder.layer." in n) and (n.endswith(".query") or n.endswith(".value")))
print(f"swin qkv linears: {n_qkv} | bert query/value linears: {n_cross_qv}")

# forward with random-ish inputs
from transformers import BertTokenizer  # noqa: E402
tok = BertTokenizer.from_pretrained("bert-base-uncased")
batch_txt = tok(["a man is falling on the street", "a person running"],
                padding="max_length", truncation=True, max_length=40, return_tensors="pt")
image = torch.randn(2, 3, 384, 384)

with torch.no_grad():
    img_embeds, img_atts = model.get_vision_embeds(image)
    txt_embeds = model.get_text_embeds(batch_txt["input_ids"], batch_txt["attention_mask"])
    img_feat, txt_feat = model.get_features(img_embeds, txt_embeds)
    cross = model.get_cross_embeds(img_embeds, img_atts,
                                   text_embeds=txt_embeds, text_atts=batch_txt["attention_mask"])
    itm = model.itm_head(cross[:, 0, :])

print("img_embeds", tuple(img_embeds.shape), "| txt_embeds", tuple(txt_embeds.shape))
print("img_feat", tuple(img_feat.shape), "| txt_feat", tuple(txt_feat.shape), "| dim ok:", img_feat.shape[-1] == 256)
print("cross", tuple(cross.shape), "| itm_logits", tuple(itm.shape))
print("cosine(img0,txt0) =", float((img_feat[0] * txt_feat[0]).sum()))
print("PROBE OK")
