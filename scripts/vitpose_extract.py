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
import time
from pathlib import Path

# Reuse the YOLO extractor's formatters so the JSON schema stays the same contract.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_pose_yolo import empty_item, to_item   # noqa: E402

_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def hf_to_arrays(pose_result, det_boxes_xyxy, det_scores):
    """HF post_process_pose_estimation per-image result (list of person dicts) ->
    (boxes_xyxy [n,4], kdata [n,17,3]) for pick_primary. PURE -> unit-testable without models.

    Tolerant to schema drift across transformers versions: keypoints may be [17,2] (+ a separate
    'scores' list) or [17,3] (conf baked in). Person boxes/scores come from the same RT-DETR
    detector used before ViTPose, matching the preprocessing extractor's primary-person rule."""
    boxes, scores, kdata = [], [], []
    for i, person in enumerate(pose_result):
        kp = person["keypoints"]
        sc = person.get("scores")
        triples, xs, ys = [], [], []
        for j in range(len(kp)):
            x, y = float(kp[j][0]), float(kp[j][1])
            c = float(sc[j]) if sc is not None else (float(kp[j][2]) if len(kp[j]) > 2 else 1.0)
            triples.append([x, y, c]); xs.append(x); ys.append(y)
        if i < len(det_boxes_xyxy):
            boxes.append([float(v) for v in det_boxes_xyxy[i]])
        else:
            boxes.append([min(xs), min(ys), max(xs), max(ys)] if xs else [0.0, 0.0, 0.0, 0.0])
        scores.append(float(det_scores[i]) if i < len(det_scores) else 0.0)
        kdata.append(triples)
    return boxes, scores, kdata


def pick_primary_by_det_score_area(boxes_xyxy, det_scores, kpts_data):
    """Match preprocessing: primary person = max(det_score * detector_bbox_area)."""
    if kpts_data is None or len(kpts_data) == 0:
        return None
    best, best_score = 0, -1.0
    for i, b in enumerate(boxes_xyxy):
        area = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
        score = (float(det_scores[i]) if i < len(det_scores) else 0.0) * max(area, 1.0)
        if score > best_score:
            best, best_score = i, score
    return [list(map(float, kp)) for kp in kpts_data[best]], [float(v) for v in boxes_xyxy[best]]


def _rows_from_args(args):
    if args.image_dir:
        files = [p for p in glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True)
                 if p.lower().endswith(_EXTS)]
        args.image_root = ""                      # rows carry absolute paths
        return [{"image": p, "image_id": Path(p).stem} for p in files]
    assert args.attr, "can --attr hoac --image-dir"
    return [json.loads(l) for l in open(args.attr, encoding="utf-8")]


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _load_items(path):
    path = Path(path)
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("items", {})


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
    ap.add_argument("--progress-every", type=int, default=50,
                    help="print progress every N images; 0 disables progress logs")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N images for smoke tests")
    ap.add_argument("--resume", action="store_true",
                    help="resume from --out or its partial JSON if present")
    ap.add_argument("--save-every", type=int, default=500,
                    help="write partial JSON every N newly processed images; 0 disables")
    ap.add_argument("--partial-out", default=None,
                    help="partial JSON path; default is <out>.partial.json")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import (AutoProcessor, RTDetrForObjectDetection,
                              VitPoseForPoseEstimation)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    # float32: ViTPose post-processing (post_dark_unbiased_data_processing -> scipy.gaussian_filter)
    # does NOT support float16 heatmaps. Both models are small -> float32 fits T4 easily.
    dt = torch.float32
    det_proc = AutoProcessor.from_pretrained(args.detector)
    det = RTDetrForObjectDetection.from_pretrained(args.detector, torch_dtype=dt).to(device).eval()
    pose_proc = AutoProcessor.from_pretrained(args.pose_model)
    pose = VitPoseForPoseEstimation.from_pretrained(args.pose_model, torch_dtype=dt).to(device).eval()

    rows = _rows_from_args(args)
    if args.limit is not None:
        rows = rows[:max(args.limit, 0)]
    print(f"images: {len(rows)} | device: {device} | detector: {args.detector} | pose: {args.pose_model}",
          flush=True)
    out_path = Path(args.out)
    partial_path = Path(args.partial_out) if args.partial_out else out_path.with_suffix(".partial.json")
    items = {}
    if args.resume:
        resume_path = out_path if out_path.exists() else partial_path
        items = _load_items(resume_path)
        if items:
            print(f"resume: loaded {len(items):,} existing items from {resume_path}", flush=True)
    ok = sum(1 for item in items.values() if item.get("status") == "ok")
    skipped = 0
    newly_processed = 0
    t0 = time.time()
    for idx, r in enumerate(rows, start=1):
        image_id = str(r["image_id"])
        if image_id in items:
            skipped += 1
            if args.progress_every and (idx % args.progress_every == 0 or idx == len(rows)):
                elapsed = max(time.time() - t0, 1e-6)
                done = skipped + newly_processed
                speed = done / elapsed
                remaining = (len(rows) - done) / max(speed, 1e-6)
                print(f"pose processed {done:,}/{len(rows):,}; skipped {skipped:,}; ok {ok:,}; "
                      f"{speed:.2f} img/s; eta {remaining / 60:.1f} min",
                      flush=True)
            continue
        p = Path(args.image_root) / r["image"] if args.image_root else Path(r["image"])
        image = Image.open(p).convert("RGB")
        w, h = image.size
        # ---- stage 1: detect persons (COCO label 0) ----
        di = det_proc(images=image, return_tensors="pt").to(device, dt)
        with torch.no_grad():
            do = det(**di)
        dres = det_proc.post_process_object_detection(
            do, target_sizes=torch.tensor([(h, w)]).to(device), threshold=args.det_threshold)[0]
        person_mask = dres["labels"] == 0
        pboxes = dres["boxes"][person_mask]          # xyxy, pixels
        pscores = dres["scores"][person_mask]
        if pboxes.numel() == 0:
            items[image_id] = empty_item(w, h)
        else:
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
            bxyxy, dscores, kdata = hf_to_arrays(pres, pboxes.detach().cpu().tolist(), pscores.detach().cpu().tolist())
            pick = pick_primary_by_det_score_area(bxyxy, dscores, kdata)
            if pick is None:
                items[image_id] = empty_item(w, h)
            else:
                kpts, box = pick
                items[image_id] = to_item(kpts, box, w, h)
                ok += 1
        newly_processed += 1
        if args.save_every and newly_processed % args.save_every == 0:
            _write_json_atomic(partial_path, {"items": items, "meta": {
                "detector": args.detector, "pose_model": args.pose_model,
                "n": len(rows), "ok": ok, "partial": True}})
        if args.progress_every and (idx % args.progress_every == 0 or idx == len(rows)):
            elapsed = max(time.time() - t0, 1e-6)
            done = skipped + newly_processed
            speed = done / elapsed
            remaining = (len(rows) - done) / max(speed, 1e-6)
            print(f"pose processed {done:,}/{len(rows):,}; skipped {skipped:,}; ok {ok:,}; "
                  f"{speed:.2f} img/s; eta {remaining / 60:.1f} min",
                  flush=True)
    _write_json_atomic(out_path, {"items": items, "meta": {
        "detector": args.detector, "pose_model": args.pose_model,
        "n": len(rows), "ok": ok, "skipped": skipped}})
    print(f"wrote {args.out}: {ok}/{len(rows)} images with a detected person "
          f"({ok / max(len(rows), 1):.0%} coverage)")


if __name__ == "__main__":
    main()
