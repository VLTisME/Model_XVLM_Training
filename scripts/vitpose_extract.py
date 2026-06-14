"""Extract COCO-17 keypoints with REAL ViTPose via HuggingFace transformers (NO mmpose/mmcv,
which clash with the pinned X-VLM env). Output schema is IDENTICAL to extract_pose_yolo.py
(it reuses the same formatters), so pose_rerank.py / kpts_of() consume it unchanged.

This is the train-consistent ("dong bo") option: training used ViTPose keypoints, so re-ranking
with ViTPose keypoints (not YOLO) removes the detector/regressor distribution shift. ViTPose is
top-down -> needs person boxes first: RT-DETR detector -> boxes -> ViTPose -> 17 keypoints.

RUN IN A SUBPROCESS with a MODERN transformers (>=4.49 has VitPoseForPoseEstimation), separate
from X-VLM's pinned old transformers -> no env conflict (same pattern as the Qwen reranker).
Needs Kaggle internet ON to pull the HF weights, or pass local paths via --detector/--pose-model.

    python scripts/vitpose_extract.py --image-dir /old-test/images --out kp.json
    python scripts/vitpose_extract.py --attr attr.json --image-root /old-test --out kp.json

Output: {"items": {"<image_id>": {"status","width","height",
                                 "instances":[{"keypoints_xyc":[[x,y,c]*17]}],
                                 "primary_bbox_norm_xyxy"}}, "meta": {...}}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

# reuse the YOLO extractor's formatters so the json schema is byte-for-byte the same contract
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_pose_yolo import empty_item, pick_primary, to_item   # noqa: E402

_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def hf_to_arrays(pose_result):
    """HF post_process_pose_estimation per-image result (list of person dicts) ->
    (boxes_xyxy [n,4], kdata [n,17,3]) for pick_primary. PURE -> unit-testable without models.

    Tolerant to schema drift across transformers versions: keypoints may be [17,2] (+ a separate
    'scores' list) or [17,3] (conf baked in). The box is derived from the KEYPOINT EXTENT, not the
    detector bbox -- it is only used to pick the largest person, and the extent is unambiguous
    across versions (sidesteps the xyxy-vs-xywh bbox ambiguity entirely)."""
    boxes, kdata = [], []
    for person in pose_result:
        kp = person["keypoints"]
        sc = person.get("scores")
        triples, xs, ys = [], [], []
        for j in range(len(kp)):
            x, y = float(kp[j][0]), float(kp[j][1])
            c = float(sc[j]) if sc is not None else (float(kp[j][2]) if len(kp[j]) > 2 else 1.0)
            triples.append([x, y, c]); xs.append(x); ys.append(y)
        boxes.append([min(xs), min(ys), max(xs), max(ys)] if xs else [0.0, 0.0, 0.0, 0.0])
        kdata.append(triples)
    return boxes, kdata


def _rows_from_args(args):
    if args.image_dir:
        files = [p for p in glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True)
                 if p.lower().endswith(_EXTS)]
        args.image_root = ""                      # rows carry absolute paths
        return [{"image": p, "image_id": Path(p).stem} for p in files]
    assert args.attr, "can --attr hoac --image-dir"
    return [json.loads(l) for l in open(args.attr, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default=None, help="attr.json (one json row per line: image, image_id)")
    ap.add_argument("--image-dir", default=None, help="instead of --attr: glob every image in a dir, image_id = file stem")
    ap.add_argument("--image-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector", default="PekingU/rtdetr_r50vd_coco_o365", help="HF person detector")
    ap.add_argument("--pose-model", default="usyd-community/vitpose-base-simple", help="HF ViTPose")
    ap.add_argument("--det-threshold", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import (AutoProcessor, RTDetrForObjectDetection,
                              VitPoseForPoseEstimation)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if device == "cuda" else torch.float32
    det_proc = AutoProcessor.from_pretrained(args.detector)
    det = RTDetrForObjectDetection.from_pretrained(args.detector, torch_dtype=dt).to(device).eval()
    pose_proc = AutoProcessor.from_pretrained(args.pose_model)
    pose = VitPoseForPoseEstimation.from_pretrained(args.pose_model, torch_dtype=dt).to(device).eval()

    rows = _rows_from_args(args)
    items, ok = {}, 0
    for r in rows:
        p = Path(args.image_root) / r["image"] if args.image_root else Path(r["image"])
        image = Image.open(p).convert("RGB")
        w, h = image.size
        # ---- stage 1: detect persons (COCO label 0) ----
        di = det_proc(images=image, return_tensors="pt").to(device, dt)
        with torch.no_grad():
            do = det(**di)
        dres = det_proc.post_process_object_detection(
            do, target_sizes=torch.tensor([(h, w)]).to(device), threshold=args.det_threshold)[0]
        pboxes = dres["boxes"][dres["labels"] == 0]          # xyxy, pixels
        if pboxes.numel() == 0:
            items[str(r["image_id"])] = empty_item(w, h); continue
        # xyxy -> xywh for the pose processor
        boxes_xywh = pboxes.detach().clone().float()
        boxes_xywh[:, 2] -= boxes_xywh[:, 0]
        boxes_xywh[:, 3] -= boxes_xywh[:, 1]
        boxes_np = boxes_xywh.cpu().numpy()
        # ---- stage 2: ViTPose on each box ----
        pi = pose_proc(image, boxes=[boxes_np], return_tensors="pt").to(device, dt)
        with torch.no_grad():
            po = pose(**pi)
        pres = pose_proc.post_process_pose_estimation(po, boxes=[boxes_np])[0]
        bxyxy, kdata = hf_to_arrays(pres)
        pick = pick_primary(bxyxy, kdata)
        if pick is None:
            items[str(r["image_id"])] = empty_item(w, h)
        else:
            kpts, box = pick
            items[str(r["image_id"])] = to_item(kpts, box, w, h)
            ok += 1
    Path(args.out).write_text(json.dumps({"items": items, "meta": {
        "detector": args.detector, "pose_model": args.pose_model, "n": len(rows), "ok": ok}}))
    print(f"wrote {args.out}: {ok}/{len(rows)} images with a detected person "
          f"({ok / max(len(rows), 1):.0%} coverage)")


if __name__ == "__main__":
    main()
