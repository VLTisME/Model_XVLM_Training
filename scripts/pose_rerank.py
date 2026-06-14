"""Pose-ON re-rank: use the TRAINED ViTPose branch (which the appearance-only X-VLM inference
DISCARDS) to re-rank each query's top-K — fixing same-scene / different-pose confusions
(rank-1 vs the GT at rank-2, both frames of one video).

The X-VLM inference loads only `backbone.*` -> the `pose.*` branch in best.pth is never used ->
the exact signal that disambiguates same-scene frames is thrown away. Here we load the pose branch,
fuse keypoints into the gallery feature (train-consistent), recompute the cosine, and re-rank the
top-K. Unlike a 1-1 Hungarian assignment (which only works without distractors), pose is a REAL
content discriminator -> it transfers to the real distractor test.

    python pose_rerank.py --best best.pth --base-ckpt xvlm_16m_base.th \
        --answer answer.txt --query-json query_text.json --query-index query_index.txt \
        --gt ground_truth.txt --image-root /kaggle/input --pose-json oldtest_pose.json \
        --topk 5 --out answer_pose.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.config import _merge, load_config              # noqa: E402
from star.data import PABDataset                          # noqa: E402
from star.inference import encode_eval_set                # noqa: E402
from star.models import STARModel                         # noqa: E402

_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def rrf(orders, k: int = 60):
    """Reciprocal-rank fusion of ranked id-lists over the same set."""
    score: dict = {}
    for order in orders:
        for rank, idx in enumerate(order):
            score[idx] = score.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: score[i], reverse=True)


def rerank_topk(full, cand_scores, k, blend=False):
    """Re-rank the first `k` of `full` by the pose-ON cosine (the train-consistent score =
    appearance + pose), keep the tail. Default is PURE pose order — RRF blend of a clean top-2
    swap is a symmetric tie and never flips rank-1<->rank-2, which is exactly what we need to fix."""
    top = full[:k]
    pose_order = sorted(top, key=lambda c: cand_scores.get(c, -1e9), reverse=True)
    new_top = rrf([top, pose_order]) if blend else pose_order
    return new_top + full[k:]


def metrics(rank_of):
    import numpy as np
    rr = np.array([1.0 / r if r else 0.0 for r in rank_of])
    valid = [r for r in rank_of if r]
    hit = lambda kk: (sum(1 for r in valid if r <= kk) / len(valid)) if valid else 0.0
    return dict(mAP=float(rr.mean()), R1=hit(1), R5=hit(5), R10=hit(10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--config", default="configs/star_v3_10k_kaggle.yaml")
    ap.add_argument("--answer", required=True)
    ap.add_argument("--query-json", required=True)
    ap.add_argument("--query-index", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--image-root", default="/kaggle/input")
    ap.add_argument("--pose-json", required=True, help="keypoints json (extract_pose_yolo / ViTPose schema)")
    ap.add_argument("--out", default="/kaggle/working/outputs/answer_pose.txt")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--blend", action="store_true",
                    help="RRF-fuse pose order with original (conservative). Default = pure pose-ON re-rank.")
    ap.add_argument("--skip-first-col", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- inputs ----
    raw = open(args.query_json, encoding="utf-8").read().strip()
    recs = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    cap_of = {str(r["query_index"]): r["caption"] for r in recs}
    qorder = [l.strip() for l in open(args.query_index, encoding="utf-8").read().strip().splitlines()]
    gts = [l.strip() for l in open(args.gt, encoding="utf-8").read().strip().splitlines()]
    ans = [l.split() for l in open(args.answer, encoding="utf-8").read().strip().splitlines()]
    if args.skip_first_col:
        ans = [t[1:] for t in ans]
    assert len(qorder) == len(gts) == len(ans), f"lech dong: {len(qorder)} {len(gts)} {len(ans)}"
    stem2path = {Path(p).stem: p for p in glob.glob(os.path.join(args.image_root, "**", "*"), recursive=True)
                 if p.lower().endswith(_EXTS)}
    pose = json.load(open(args.pose_json, encoding="utf-8")).get("items", {})

    def kpts_of(iid):
        it = pose.get(str(iid))
        if not it or it.get("status") != "ok" or not it.get("instances"):
            return [0.0] * 51                         # zero-fill -> pose applied uniformly
        W, H = it.get("width", 384), it.get("height", 384)
        flat = []
        for x, y, c in it["instances"][0]["keypoints_xyc"]:
            flat += [x / W, y / H, c]
        return flat if len(flat) == 51 else [0.0] * 51

    # ---- model WITH pose branch (the part the X-VLM inference dropped) ----
    cfg = load_config(args.config)
    ck = torch.load(args.best, map_location="cpu", weights_only=False)
    _merge(cfg.model, ((ck.get("extra") or {}).get("cfg") or {}).get("model", {}))
    cfg.model.checkpoint = args.base_ckpt
    model = STARModel(cfg).to(device).eval()
    msg = model.load_state_dict(ck["model"], strict=False)
    n_pose = sum(1 for k in ck["model"] if k.startswith("pose."))
    print(f"loaded best.pth: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} "
          f"| pose tensors loaded={n_pose} | pose branch={'ON' if model.pose is not None else 'OFF'}")
    assert model.pose is not None, "ckpt khong co pose branch -> khong the pose-rerank"
    del ck

    # ---- manifest: gallery == query set (no-distractor); every image is a query+candidate ----
    import pandas as pd
    rows = []
    for i, qid in enumerate(qorder):
        p = stem2path.get(qid)
        rows.append(dict(image_path=p if p else "", caption=cap_of.get(qid, ""), split="valb",
                         sequence_id=f"o{qid}", scene=f"o{qid}", action="q",
                         image_id=str(qid), bbox=None, keypoints=kpts_of(qid)))
    import tempfile
    mani = os.path.join(tempfile.gettempdir(), "_pose_rerank.parquet")
    pd.DataFrame(rows).to_parquet(mani, index=False)
    ds = PABDataset(mani, args.image_root, model.backbone.tokenizer, split="valb",
                    image_size=cfg.data.image_size, max_token=cfg.data.max_token, train=False)

    # ---- pose-ON cosine (encode_eval_set fuses pose because model.pose set + keypoints present) ----
    enc = encode_eval_set(model, ds, device, batch_size=64, num_workers=2)
    sim = enc["txt_feats"] @ enc["gallery_feats"].t()             # [Q, G] POSE-ON
    gid = {g: j for j, g in enumerate(enc["gallery_ids"])}
    qpos = {str(qorder[i]): i for i in range(len(qorder))}        # answer row -> ... but enc query order
    # enc query order: encode_eval_set yields queries in manifest order = qorder order -> row i aligns
    print(f"encoded: gallery={sim.size(1)} queries={sim.size(0)} (pose-ON cosine)")

    def rank_in(order, gt):
        return order.index(gt) + 1 if gt in order else 0

    before, after = [], []
    out_lines = []
    for i in range(len(ans)):
        full = ans[i]
        cand_scores = {c: float(sim[i, gid[c]]) for c in full[:args.topk] if c in gid}
        new_full = rerank_topk(full, cand_scores, args.topk, blend=args.blend)
        before.append(rank_in(full, gts[i]))
        after.append(rank_in(new_full, gts[i]))
        out_lines.append(" ".join(new_full))

    B, A = metrics(before), metrics(after)
    print(f"\n{'':8s}{'mAP':>9}{'R@1':>9}{'R@5':>9}{'R@10':>9}")
    print(f"{'X-VLM':8s}{B['mAP']:9.4f}{B['R1']:9.4f}{B['R5']:9.4f}{B['R10']:9.4f}")
    print(f"{'+pose':8s}{A['mAP']:9.4f}{A['R1']:9.4f}{A['R5']:9.4f}{A['R10']:9.4f}")
    print(f"{'delta':8s}{A['mAP']-B['mAP']:+9.4f}{A['R1']-B['R1']:+9.4f}{A['R5']-B['R5']:+9.4f}{A['R10']-B['R10']:+9.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out_lines) + "\n")
    (Path(args.out).parent / "pose_metrics.json").write_text(json.dumps(
        {"before": B, "after": A, "ranks_before": before, "ranks_after": after}))
    print("\nwrote", args.out, "+ pose_metrics.json")


if __name__ == "__main__":
    main()
