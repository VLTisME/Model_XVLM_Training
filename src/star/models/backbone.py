"""Backbone wrapper + a runnable dummy fallback.

See analyze.md (Backbone interface). The rest of the codebase depends ONLY on:
  - backbone.tokenizer                                       (HF-like)
  - backbone.encode_image(image)            -> (img_embeds [B,Ni,H], img_feat [B,d])
  - backbone.encode_text(ids, mask)         -> (txt_embeds [B,Nt,H], txt_feat [B,d])
  - backbone.itm_logits(img_embeds, txt_embeds, txt_mask)   -> [P,2]
  - backbone.setup_finetuning(cfg)          -> int  (inject LoRA on image+cross, freeze text)

`_DummyXVLM` is a small but REAL trainable model so `pytest` and `--overfit-one-batch`
work with zero external downloads. Swap in real X-VLM by implementing the same methods
(see `XVLMBackbone`). MLM was removed from the training objective (plan), so there is no mlm head.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .lora import inject_lora

# substrings identifying TEXT-tower params in the dummy net (frozen when lora_freeze_text)
_DUMMY_TEXT_KEYS = ("tok_embed", "txt_pos", "text_self", "txt_proj")


@dataclass
class BackboneOut:
    img_feat: Tensor      # [B, d] L2-normalized
    txt_feat: Tensor      # [B, d] L2-normalized
    img_embeds: Tensor    # [B, Ni, H]
    txt_embeds: Tensor    # [B, Nt, H]
    txt_mask: Tensor      # [B, Nt]


# ----------------------------------------------------------------- minimal HF-like tokenizer
class SimpleTokenizer:
    """Whitespace+hash tokenizer mimicking the HF interface (offline, for the dummy backbone)."""

    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    unk_token_id = 4

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.all_special_ids = [self.pad_token_id, self.cls_token_id, self.sep_token_id,
                                self.mask_token_id, self.unk_token_id]

    def _tok(self, w: str) -> int:
        return 5 + (hash(w) % (self.vocab_size - 5))

    def __call__(self, text, padding="max_length", truncation=True, max_length=100, return_tensors="pt"):
        words = str(text).lower().split()[: max_length - 2]
        ids = [self.cls_token_id] + [self._tok(w) for w in words] + [self.sep_token_id]
        attn = [1] * len(ids)
        if padding == "max_length":
            pad = max_length - len(ids)
            ids = ids + [self.pad_token_id] * pad
            attn = attn + [0] * pad
        ids, attn = ids[:max_length], attn[:max_length]
        t_ids = torch.tensor([ids], dtype=torch.long)
        t_attn = torch.tensor([attn], dtype=torch.long)
        return {"input_ids": t_ids, "attention_mask": t_attn}


# ----------------------------------------------------------------- dummy but real model
class _DummyXVLM(nn.Module):
    def __init__(self, hidden: int = 256, embed: int = 256, vocab: int = 1000, img_tokens: int = 16):
        super().__init__()
        self.hidden = hidden
        self.patch = nn.Conv2d(3, hidden, kernel_size=32, stride=32)   # 384/32 -> 12x12 patches
        self.img_pos = nn.Parameter(torch.zeros(1, 145, hidden))       # up to 144 patches + 1
        self.tok_embed = nn.Embedding(vocab, hidden, padding_idx=0)
        self.txt_pos = nn.Parameter(torch.zeros(1, 128, hidden))
        enc_layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2,
                                               batch_first=True, dropout=0.0)
        self.text_self = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.cross_ffn = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden))
        self.img_proj = nn.Linear(hidden, embed)
        self.txt_proj = nn.Linear(hidden, embed)
        self.itm_head = nn.Linear(hidden, 2)

    def encode_image(self, image: Tensor):
        x = self.patch(image)                       # [B, H, h, w]
        b, h, hh, ww = x.shape
        x = x.flatten(2).transpose(1, 2)            # [B, Ni, H]
        x = x + self.img_pos[:, : x.size(1)]
        feat = F.normalize(self.img_proj(x.mean(dim=1)), dim=-1)
        return x, feat

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor):
        e = self.tok_embed(input_ids) + self.txt_pos[:, : input_ids.size(1)]
        pad = attention_mask == 0
        h = self.text_self(e, src_key_padding_mask=pad)
        feat = F.normalize(self.txt_proj(h[:, 0]), dim=-1)
        return h, feat

    def _cross(self, img_embeds: Tensor, txt_embeds: Tensor, txt_mask: Tensor) -> Tensor:
        attn_out, _ = self.cross_attn(txt_embeds, img_embeds, img_embeds, need_weights=False)
        fused = txt_embeds + attn_out
        return fused + self.cross_ffn(fused)        # [P, Nt, H]

    def itm_logits(self, img_embeds, txt_embeds, txt_mask):
        fused = self._cross(img_embeds, txt_embeds, txt_mask)
        return self.itm_head(fused[:, 0])           # [P, 2] from [CLS]

    def cross_feature(self, img_embeds, txt_embeds, txt_mask):
        return self._cross(img_embeds, txt_embeds, txt_mask)[:, 0]   # [P, H] fused [CLS]


class DummyBackbone(nn.Module):
    """Adapter exposing the stable interface around _DummyXVLM."""

    def __init__(self, embed_dim: int = 256, vocab: int = 1000):
        super().__init__()
        self.tokenizer = SimpleTokenizer(vocab_size=vocab)
        self.net = _DummyXVLM(embed=embed_dim, vocab=vocab)

    def encode_image(self, image):
        return self.net.encode_image(image)

    def encode_text(self, ids, mask):
        return self.net.encode_text(ids, mask)

    def itm_logits(self, img_embeds, txt_embeds, txt_mask):
        return self.net.itm_logits(img_embeds, txt_embeds, txt_mask)

    def cross_feature(self, img_embeds, txt_embeds, txt_mask):
        return self.net.cross_feature(img_embeds, txt_embeds, txt_mask)

    def setup_finetuning(self, cfg) -> int:
        """Inject LoRA (image + cross only) and freeze the text tower per the plan.

        Returns the number of LoRA layers injected. The dummy net has no attention Q/V Linear
        modules, so n_lora is 0 here (the whole net trains, minus the frozen text tower) — which
        is exactly what we want for fast CPU tests. The real XVLMBackbone overrides this.
        """
        n_lora = 0
        if cfg.model.lora_enabled:
            exclude = _DUMMY_TEXT_KEYS if cfg.model.lora_freeze_text else ()
            n_lora = inject_lora(
                self.net, targets=tuple(cfg.model.lora_targets),
                r=cfg.model.lora_r, alpha=cfg.model.lora_alpha,
                dropout=cfg.model.lora_dropout, exclude=exclude,
            )
        if cfg.model.lora_freeze_text:
            for name, p in self.net.named_parameters():
                if any(k in name for k in _DUMMY_TEXT_KEYS):
                    p.requires_grad_(False)
        return n_lora


# text-tower module names to skip for LoRA / freeze on the real X-VLM (BERT layers 0-5 = text)
_XVLM_TEXT_EXCLUDE = ("embeddings", "layer.0.", "layer.1.", "layer.2.", "layer.3.", "layer.4.", "layer.5.")


class XVLMBackbone(nn.Module):
    """Real X-VLM wrapper (validated against third_party/X-VLM + the 16M checkpoint).

    Maps X-VLM's API to the interface STARModel needs:
      encode_image  -> get_vision_embeds + get_features        (img_embeds [B,145,1024], f_V [B,256])
      encode_text   -> get_text_embeds  + get_features         (txt_embeds [B,L,768],  f_T [B,256])
      itm_logits    -> get_cross_embeds[:,0] -> itm_head       ([P,2])
      .temp         -> the model's PRETRAINED temperature (review fix #6)
    setup_finetuning: LoRA on Swin `qkv` (image) + BERT `query/value` of fusion layers 6-11 (cross);
    text layers 0-5 + embeddings + text_proj are left untouched -> frozen by mark_only_lora_trainable.

    X-VLM (transformers==4.12.5) runs in a separate pinned venv; the import is LAZY so the main
    environment (tests, dummy backbone) is unaffected.
    """

    def __init__(self, cfg):
        super().__init__()
        import sys
        from pathlib import Path

        repo = Path(getattr(cfg.model, "xvlm_repo", "") or
                    (Path(__file__).resolve().parents[3] / "third_party" / "X-VLM"))
        required_source = repo / "models" / "model_retrieval.py"
        if not required_source.exists():
            raise FileNotFoundError(
                "X-VLM source code is missing. Expected:\n"
                f"  {required_source}\n"
                "Clone it with:\n"
                f"  git clone --depth 1 https://github.com/zengyan-97/X-VLM.git {repo}\n"
                "or pass an existing checkout with:\n"
                "  --set model.xvlm_repo=/absolute/path/to/X-VLM"
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        ckpt = Path(cfg.model.checkpoint)
        if not ckpt.is_absolute():
            ckpt = Path(__file__).resolve().parents[3] / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"X-VLM checkpoint does not exist: {ckpt}")

        try:
            from models.model_retrieval import XVLM  # noqa: E402  (X-VLM repo)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Could not import X-VLM from {repo}. "
                "Verify that this directory contains models/model_retrieval.py."
            ) from exc
        from transformers import BertTokenizer   # noqa: E402

        self._build_cfg = {
            "use_swin": True, "use_clip_vit": False, "use_roberta": False,
            "image_res": cfg.data.image_size, "patch_size": 32,
            "vision_config": str(repo / "configs" / "config_swinB_384.json"),
            "text_config": str(repo / "configs" / "config_bert.json"),
            "text_encoder": "bert-base-uncased",
            "embed_dim": cfg.model.embed_dim, "temp": cfg.loss.itc_temp_init,
        }
        self.model = XVLM(config=self._build_cfg)
        self.model.load_pretrained(str(ckpt), self._build_cfg, is_eval=False)
        if cfg.train.grad_checkpointing:
            vision_setter = getattr(self.model.vision_encoder, "set_grad_checkpointing", None)
            if callable(vision_setter):
                vision_setter(True)
            text_setter = getattr(self.model.text_encoder, "gradient_checkpointing_enable", None)
            if callable(text_setter):
                try:
                    text_setter()
                except (ValueError, NotImplementedError):
                    # X-VLM is pinned to transformers 4.12.5. Its BertModel exposes this
                    # method through PreTrainedModel but declares checkpointing unsupported.
                    # Keep Swin checkpointing enabled and let the old BERT run normally.
                    pass
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.temp = self.model.temp     # share the PRETRAINED temperature with ITCLoss (fix #6)

    # ----- interface used by STARModel -----
    def encode_image(self, image):
        img_embeds, _ = self.model.get_vision_embeds(image)
        img_feat = self.model.get_features(image_embeds=img_embeds)
        return img_embeds, img_feat

    def encode_text(self, ids, mask):
        txt_embeds = self.model.get_text_embeds(ids, mask)
        txt_feat = self.model.get_features(text_embeds=txt_embeds)
        return txt_embeds, txt_feat

    def itm_logits(self, img_embeds, txt_embeds, txt_mask):
        img_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long, device=img_embeds.device)
        cross = self.model.get_cross_embeds(img_embeds, img_atts, text_embeds=txt_embeds, text_atts=txt_mask)
        return self.model.itm_head(cross[:, 0, :])

    def cross_feature(self, img_embeds, txt_embeds, txt_mask):
        img_atts = torch.ones(img_embeds.size()[:-1], dtype=torch.long, device=img_embeds.device)
        cross = self.model.get_cross_embeds(img_embeds, img_atts, text_embeds=txt_embeds, text_atts=txt_mask)
        return cross[:, 0, :]                          # [P, H] fused [CLS] (the vector ITM head reads)

    def setup_finetuning(self, cfg) -> int:
        from .lora import inject_lora
        if not cfg.model.lora_enabled:
            return 0
        kw = dict(r=cfg.model.lora_r, alpha=cfg.model.lora_alpha, dropout=cfg.model.lora_dropout)
        n = inject_lora(self.model.vision_encoder, targets=("qkv",), **kw)          # image (Swin)
        exclude = _XVLM_TEXT_EXCLUDE if cfg.model.lora_freeze_text else ()
        n += inject_lora(self.model.text_encoder, targets=tuple(cfg.model.lora_targets),
                         exclude=exclude, **kw)                                     # cross layers 6-11
        return n


def build_backbone(cfg) -> nn.Module:
    """Factory. Falls back to the dummy backbone if real X-VLM is not configured."""
    if cfg.model.backbone == "xvlm" and cfg.model.checkpoint:
        return XVLMBackbone(cfg)
    return DummyBackbone(embed_dim=cfg.model.embed_dim)
