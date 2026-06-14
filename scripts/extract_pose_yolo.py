"""Extract COCO-17 keypoints with YOLOv8-pose, in the SAME json schema as the team's
`train_10k_hard_vitpose.json`, so the eval notebooks' `kpts_of()` can consume it unchanged.

    python scripts/extract_pose_yolo.py --attr attr.json --image-root /old-test --out kp.json

Output: {"items": {"<image_id>": {"status","width","height",
                                  "instances":[{"keypoints_xyc":[[x,y,c]*17]}],
                                  "primary_bbox_norm_xyxy"}},
         "meta": {...}}

A lightweight, pip-installable stand-in for ViTPose (mmpose/mmcv, which clashes with the pinned
X-VLM env). COCO-17 ordering matches ViTPose, so the pose branch reads it the same way; only the
detector/regressor differs (minor distribution shift vs the keypoints used in training).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def to_item(kpts_xyc, box_xyxy, width: int, height: int, status: str = "ok") -> dict:
    """Pure formatter (no YOLO needed -> unit-testable).

    Args:
        kpts_xyc: iterable of 17 [x, y, conf] in PIXEL coords.
        box_xyxy: [x1, y1, x2, y2] in pixels, or None.
    """
    item = {
        "status": status,
        "width": int(width),
        "height": int(height),
        "instances": [{"keypoints_xyc": [[float(x), float(y), float(c)] for x, y, c in kpts_xyc]}],
    }
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        item["primary_bbox_norm_xyxy"] = [x1 / width, y1 / height, x2 / width, y2 / height]
    return item


def empty_item(width: int, height: int) -> dict:
    """No person detected -> consumer's `kpts_of` sees status != 'ok' and zero-fills."""
    return {"status": "empty", "width": int(width), "height": int(height), "instances": []}


def pick_primary(boxes_xyxy, kpts_data):
    """Largest-area detected person -> (kpts_xyc [17,3] list, box_xyxy list); None if no detection.

    boxes_xyxy: [n,4] array-like; kpts_data: [n,17,3] array-like. Kept array-agnostic for testing.
    """
    if kpts_data is None or len(kpts_data) == 0:
        return None
    best, best_area = 0, -1.0
    for i, b in enumerate(boxes_xyxy):
        area = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
        if area > best_area:
            best, best_area = i, area
    return [list(map(float, kp)) for kp in kpts_data[best]], [float(v) for v in boxes_xyxy[best]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", default=None, help="attr.json (one json row per line: image, image_id)")
    ap.add_argument("--image-dir", default=None, help="instead of --attr: glob every image in a dir, image_id = file stem")
    ap.add_argument("--image-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8x-pose.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.image_dir:
        files = [p for p in glob.glob(os.path.join(args.image_dir, "**", "*"), recursive=True)
                 if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]
        rows = [{"image": p, "image_id": Path(p).stem} for p in files]
        args.image_root = ""        # rows carry absolute paths
    else:
        assert args.attr, "can --attr hoac --image-dir"
        rows = [json.loads(l) for l in open(args.attr, encoding="utf-8")]
    items, ok = {}, 0
    for r in rows:
        p = Path(args.image_root) / r["image"] if args.image_root else Path(r["image"])
        with Image.open(p) as im:
            w, h = im.size
        res = model.predict(str(p), verbose=False, device=args.device)[0]
        kdata = None if res.keypoints is None else res.keypoints.data.cpu().numpy()
        bdata = None if res.boxes is None else res.boxes.xyxy.cpu().numpy()
        pick = pick_primary(bdata, kdata) if (kdata is not None and bdata is not None) else None
        if pick is None:
            items[str(r["image_id"])] = empty_item(w, h)
        else:
            kpts, box = pick
            items[str(r["image_id"])] = to_item(kpts, box, w, h)
            ok += 1
    Path(args.out).write_text(json.dumps(
        {"items": items, "meta": {"model": args.model, "n": len(rows), "ok": ok}}))
    print(f"wrote {args.out}: {ok}/{len(rows)} images with a detected person "
          f"({ok / max(len(rows), 1):.0%} coverage)")


if __name__ == "__main__":
    main()
