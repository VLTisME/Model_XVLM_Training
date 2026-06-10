"""STARModel — assembles the backbone (+LoRA on image+cross, frozen text, +pose) and computes
the multi-task loss exactly as in the plan:

    L = w_itc * ITC + lambda_itm * ITM(hard-neg) + lambda_smooth_ap * Smooth-AP

Notes vs the earlier draft (per the annotated plan):
  - MLM head + loss REMOVED.
  - Text encoder is FROZEN (no LoRA); only the image encoder + cross-encoder are adapted.
  - Pose branch is fused into the IMAGE-encoder branch (affects f_V) with NO separate pose loss
    (it is trained through the ITC/Smooth-AP gradient on f_V).
See analyze.md for the math.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..losses import ITCLoss, ITMLoss, SmoothAPLoss, build_itm_pairs
from .backbone import build_backbone
from .lora import count_trainable, mark_only_lora_trainable
from .pose import PoseBranch


class STARModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)

        # LoRA on image+cross only + freeze text tower (backbone owns the module layout)
        self.n_lora = self.backbone.setup_finetuning(cfg)

        # pose branch fused into the image branch (toggle), no separate loss
        self.pose = PoseBranch(cfg.model.embed_dim, hidden=cfg.model.pose_hidden) if cfg.model.pose_enabled else None

        # losses. Review fix #6: if the backbone exposes a pretrained `temp` (real X-VLM), reuse it
        # instead of creating a fresh one (the dummy backbone has none -> ITC owns its temp).
        self.itc = ITCLoss(temp_init=cfg.loss.itc_temp_init,
                           external_temp=getattr(self.backbone, "temp", None))
        self.smap = SmoothAPLoss(tau=cfg.loss.smooth_ap_temp)
        self.itm = ITMLoss()

        # if real LoRA was injected, train only LoRA + task heads (image proj, ITM head, pose, temp).
        # NOTE: txt_proj is intentionally absent -> text side stays frozen.
        if self.n_lora > 0:
            mark_only_lora_trainable(
                self, train_heads=("itm_head", "img_proj", "vision_proj", "pose", "temp", "gate"),
            )

    # ------------------------------------------------------------------ info
    def trainable_summary(self) -> str:
        tr, tot = count_trainable(self)
        return f"trainable={tr/1e6:.2f}M / total={tot/1e6:.2f}M ({100*tr/tot:.1f}%) | lora_layers={self.n_lora}"

    # ------------------------------------------------------------------ forward / losses
    def forward(self, batch: dict, step: int = 0) -> dict[str, Tensor]:
        image, ids, mask = batch["image"], batch["input_ids"], batch["attention_mask"]
        inst = batch.get("instance_id")

        img_embeds, img_feat = self.backbone.encode_image(image)
        txt_embeds, txt_feat = self.backbone.encode_text(ids, mask)
        if self.pose is not None and "keypoints" in batch:
            img_feat = self.pose(img_feat, batch["keypoints"])     # fuse pose into the image branch

        n = img_feat.size(0)
        device = img_feat.device
        if inst is None:
            inst = torch.arange(n, device=device)

        # ---- ITC (all_gather across GPUs, identity soft targets) ----
        loss_itc = self.itc(img_feat, txt_feat, ids=inst)

        # ---- Smooth-AP (relevance = same-instance) ----
        relevance = (inst[:, None] == inst[None, :]).float()
        loss_smap = self.smap(img_feat, txt_feat, relevance)

        # ---- ITM with hard negatives ----
        # Review fix #1: sample negatives from softmax(sim / temp) like X-VLM (peaked on the
        # HARDEST negatives), not softmax(cos / 0.5). temp is already clamped by the ITC call above.
        with torch.no_grad():
            temp = self.itc.temp.detach().clamp(min=1e-3)
            sim_i2t = (img_feat @ txt_feat.t()) / temp
            forbid = inst[:, None] == inst[None, :]          # same instance => not a negative
        pairs = build_itm_pairs(sim_i2t, dup_mask=forbid, temperature=1.0)
        itm_logits = self.backbone.itm_logits(
            img_embeds[pairs["img_idx"]], txt_embeds[pairs["txt_idx"]], mask[pairs["txt_idx"]]
        )
        loss_itm = self.itm(itm_logits, pairs["label"])

        # ---- total: L = w_itc*ITC + lambda_itm*ITM + lambda_smooth_ap*SmoothAP ----
        c = self.cfg.loss
        total = c.w_itc * loss_itc + c.lambda_itm * loss_itm + c.lambda_smooth_ap * loss_smap
        return {
            "loss": total,
            "loss_itc": loss_itc.detach(),
            "loss_itm": loss_itm.detach(),
            "loss_smap": loss_smap.detach(),
        }

    @torch.no_grad()
    def encode_for_eval(self, image, ids, mask):
        """Return (img_feat, txt_feat) for retrieval evaluation (no augmentation)."""
        _, img_feat = self.backbone.encode_image(image)
        _, txt_feat = self.backbone.encode_text(ids, mask)
        return img_feat, txt_feat
