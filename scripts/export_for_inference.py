"""Export a trained STAR checkpoint (best.pth) into clean pieces for the INFERENCE team.

Usage:
    python scripts/export_for_inference.py --ckpt run10k/best.pth --out export_infer/

Input : best.pth saved by the trainer — STARModel state_dict with UNMERGED LoRA
        (every adapted Linear is stored as `<name>.base.weight` + `<name>.lora_A/B`).
Output: <out>/
    xvlm_merged.th    X-VLM-NATIVE state dict: LoRA merged into the base weights
                      (W = W0 + alpha/r * B@A), keys exactly like the official X-VLM:
                        vision_encoder.*   -> Swin-B image encoder
                        text_encoder.embeddings.* + encoder.layer.0-5.*  -> BERT text encoder
                        text_encoder.encoder.layer.6-11.*                -> cross-encoder
                        vision_proj.* / text_proj.*                      -> ITC projections
                        itm_head.*                                       -> ITM head (re-rank!)
                        temp                                             -> learned temperature
                      => loadable into a vanilla `models.model_retrieval.XVLM` (strict).
    pose_branch.pth   Pose-branch weights (encoder/proj/gate/norm). REQUIRED at inference if
                      the run had pose_enabled=true: gallery f_V must be fused with ViTPose
                      keypoints exactly like training (see export_info.json).
    export_info.json  Run config + component inventory + inference notes.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch

BB = "backbone.model."          # prefix of the wrapped X-VLM inside STARModel


def merge_lora_state_dict(sd: dict, scaling: float) -> dict:
    """Merge `<n>.base.weight + scaling * lora_B @ lora_A` -> `<n>.weight`; drop adapter keys."""
    out, merged = {}, 0
    for k, v in sd.items():
        if k.endswith(".lora_A") or k.endswith(".lora_B"):
            continue                                    # folded below
        if k.endswith(".base.weight"):
            stem = k[: -len(".base.weight")]
            a, b = sd.get(f"{stem}.lora_A"), sd.get(f"{stem}.lora_B")
            w = v + scaling * (b.float() @ a.float()).to(v.dtype) if a is not None and b is not None else v
            out[stem + ".weight"] = w
            merged += int(a is not None)
        elif k.endswith(".base.bias"):
            out[k[: -len(".base.bias")] + ".bias"] = v   # bias is never LoRA-adapted
        else:
            out[k] = v
    return out, merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pth from the trainer")
    ap.add_argument("--out", default="export_infer")
    args = ap.parse_args()

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd, extra = raw["model"], raw.get("extra", {})
    cfg = extra.get("cfg", {})
    mc = cfg.get("model", {})
    scaling = mc.get("lora_alpha", 32) / mc.get("lora_r", 16)

    sd, n_merged = merge_lora_state_dict(sd, scaling)

    xvlm, pose, skipped = {}, {}, []
    for k, v in sd.items():
        if k.startswith(BB):
            xvlm[k[len(BB):]] = v                       # -> native X-VLM keys
        elif k.startswith("pose."):
            pose[k[len("pose."):]] = v
        else:
            skipped.append(k)                           # e.g. duplicate alias backbone.temp

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(xvlm, out / "xvlm_merged.th")
    if pose:
        torch.save(pose, out / "pose_branch.pth")

    def count(prefix):
        return sum(1 for k in xvlm if k.startswith(prefix))

    info = {
        "source_ckpt": str(args.ckpt),
        "best_metric_mAP": raw.get("best_metric"),
        "val_report": extra.get("report"),
        "lora_merged_layers": n_merged,
        "lora_scaling_alpha_over_r": scaling,
        "components": {
            "vision_encoder (Swin-B image)": count("vision_encoder."),
            "text_encoder BERT layers 0-5 (frozen text)": sum(
                1 for k in xvlm if any(f"text_encoder.encoder.layer.{i}." in k for i in range(6))),
            "cross-encoder BERT layers 6-11": sum(
                1 for k in xvlm if any(f"text_encoder.encoder.layer.{i}." in k for i in range(6, 12))),
            "vision_proj/text_proj (ITC)": count("vision_proj.") + count("text_proj."),
            "itm_head (use for cross-encoder re-rank)": count("itm_head."),
            "pose_branch (separate file)": len(pose),
        },
        "skipped_duplicate_keys": skipped,
        "run_config": cfg,
        "inference_notes": [
            "Load xvlm_merged.th into vanilla X-VLM (models/model_retrieval.py XVLM) — LoRA is already merged, no adapter code needed.",
            "Stage-1 retrieve: f_V = get_features(get_vision_embeds(img)); f_T = get_features(text_embeds=get_text_embeds(ids, mask)); score = f_T @ f_V.T",
            "POSE: if run_config.model.pose_enabled, gallery f_V MUST be fused with ViTPose keypoints via pose_branch.pth (f_V' = LayerNorm(f_V + sigmoid(gate) * proj(encoder(kpts)))) — same as training, else train/eval mismatch.",
            "Images: GLOBAL full 384x384 (LHP is train-time augmentation only).",
            "Stage-2 re-rank top-K: cross = get_cross_embeds(img_embeds, img_atts, text_embeds=txt_embeds, text_atts=mask); score = softmax(itm_head(cross[:,0]), -1)[:, 1].",
            "Tokenizer: bert-base-uncased, max_length=100.",
        ],
    }
    (out / "export_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    print(f"[export] xvlm_merged.th  : {len(xvlm)} tensors ({n_merged} LoRA layers merged)")
    print(f"[export] pose_branch.pth : {len(pose)} tensors")
    print(f"[export] export_info.json written -> {out}")


if __name__ == "__main__":
    main()
