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

from ..losses import (
    ITCLoss,
    ITMLoss,
    SmoothAPLoss,
    build_explicit_itm_pairs,
    build_itm_pairs,
)
from ..losses.weighting import build_weighter
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

        # multi-task weighting: fixed (default) | uncertainty | dwa  (analyze.md §14)
        self.weighter = build_weighter(cfg.loss)

        # if real LoRA was injected, train only LoRA + task heads (image proj, ITM head, pose, temp).
        # NOTE: txt_proj is intentionally absent -> text side stays frozen.
        if self.n_lora > 0:
            mark_only_lora_trainable(
                self, train_heads=("itm_head", "img_proj", "vision_proj", "pose", "temp", "gate",
                                   "weighter"),
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
        partner = batch.get("partner_index")
        if partner is not None and bool((partner >= 0).all()):
            pairs = build_explicit_itm_pairs(partner)
        else:
            pairs = build_itm_pairs(sim_i2t, dup_mask=forbid, temperature=1.0)
        itm_logits = self.backbone.itm_logits(
            img_embeds[pairs["img_idx"]], txt_embeds[pairs["txt_idx"]], mask[pairs["txt_idx"]]
        )
        loss_itm = self.itm(itm_logits, pairs["label"])

        with torch.no_grad():
            raw_sim = img_feat @ txt_feat.t()
            diag = torch.arange(n, device=device)
            positive_similarity = raw_sim.diag().mean()
            if partner is not None and bool((partner >= 0).all()):
                paired_hard_similarity = raw_sim[diag, partner].mean()
            else:
                paired_hard_similarity = raw_sim.masked_fill(forbid, -1).max(dim=1).values.mean()
            random_similarity = (
                raw_sim[diag, torch.roll(diag, 1)].mean()
                if n > 1
                else raw_sim.new_zeros(())
            )
            predicted = itm_logits.argmax(dim=1)
            itm_positive_accuracy = (predicted[:n] == 1).float().mean()
            itm_hard_text_accuracy = (predicted[n : 2 * n] == 0).float().mean()
            itm_hard_image_accuracy = (predicted[2 * n : 3 * n] == 0).float().mean()

        # ---- total: weighter combines the tasks (fixed: w_itc*ITC + λ1*ITM + λ2*SmoothAP) ----
        total = self.weighter({"itc": loss_itc, "itm": loss_itm, "smap": loss_smap})
        # components are returned NON-detached so the trainer's per-loss grad-norm diagnostic
        # (train.grad_norm_every) can backprop through them; harmless for logging (.item()).
        return {
            "loss": total,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_smap": loss_smap,
            "positive_similarity": positive_similarity,
            "paired_hard_similarity": paired_hard_similarity,
            "random_negative_similarity": random_similarity,
            "itm_positive_accuracy": itm_positive_accuracy,
            "itm_hard_text_accuracy": itm_hard_text_accuracy,
            "itm_hard_image_accuracy": itm_hard_image_accuracy,
            "temperature": self.itc.temp.detach(),
        }

    @torch.no_grad()
    def encode_for_eval(self, image, ids, mask, keypoints=None):
        """Return (img_feat, txt_feat) for retrieval evaluation (no augmentation).

        If the pose branch is enabled, keypoints MUST be fused here too — otherwise the
        eval embedding space differs from the trained one (train/eval mismatch).
        """
        _, img_feat = self.backbone.encode_image(image)
        if self.pose is not None and keypoints is not None:
            img_feat = self.pose(img_feat, keypoints)
        _, txt_feat = self.backbone.encode_text(ids, mask)
        return img_feat, txt_feat
