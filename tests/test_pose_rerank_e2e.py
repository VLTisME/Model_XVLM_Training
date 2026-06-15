"""End-to-end smoke of scripts/pose_rerank.py on the DUMMY backbone (no X-VLM / no downloads):
builds a pose-ON checkpoint, tiny images + query/answer/gt files + a pose.json, runs the script
as a subprocess, and asserts it produces answer_pose.txt + pose_metrics.json with the right shape.

This exercises the WHOLE main() flow that the unit tests can't: manifest build -> PABDataset ->
encode_eval_set (pose fused) -> top-K re-rank -> metrics -> file writes."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pandas")
pytest.importorskip("PIL")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _kpts17():
    return [[float(i) * 5 + 10, float(i) * 4 + 12, 0.9] for i in range(17)]


def test_pose_rerank_end_to_end(tmp_path):
    sys.path.insert(0, str(SRC))
    from PIL import Image
    from star.config import Config, to_dict
    from star.models import STARModel

    ids = ["AAA", "BBB", "CCC", "DDD"]

    # ---- pose-ON dummy checkpoint (embedded cfg forces backbone=dummy on reload) ----
    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.checkpoint = None
    cfg.model.embed_dim = 64
    cfg.model.lora_enabled = False
    cfg.model.pose_enabled = True
    model = STARModel(cfg)
    ckpt = tmp_path / "best.pth"
    torch.save({"model": model.state_dict(), "step": 1, "best_metric": 0.5,
                "extra": {"cfg": to_dict(cfg)}}, ckpt)

    # ---- tiny images (image_root globbed by stem) ----
    img_root = tmp_path / "gallery"
    img_root.mkdir()
    for iid in ids:
        Image.new("RGB", (32, 32), (123, 116, 104)).save(img_root / f"{iid}.jpg")

    # ---- query / answer / gt files ----
    (tmp_path / "query_text.json").write_text(
        "\n".join(json.dumps({"query_index": iid, "caption": f"a person number {k}"})
                  for k, iid in enumerate(ids)), encoding="utf-8")
    (tmp_path / "query_index.txt").write_text("\n".join(ids), encoding="utf-8")
    (tmp_path / "ground_truth.txt").write_text("\n".join(ids), encoding="utf-8")   # GT = same-row id
    # answer.txt: GT deliberately at rank-2 for each query (so re-rank has something to move)
    ans = []
    for i, iid in enumerate(ids):
        other = ids[(i + 1) % len(ids)]
        rest = [x for x in ids if x not in (iid, other)]
        ans.append(" ".join([other, iid] + rest))
    (tmp_path / "answer.txt").write_text("\n".join(ans), encoding="utf-8")

    # ---- pose.json (schema = extract_pose_yolo / vitpose_extract output) ----
    items = {iid: {"status": "ok", "width": 32, "height": 32,
                   "instances": [{"keypoints_xyc": _kpts17()}]} for iid in ids}
    (tmp_path / "pose.json").write_text(json.dumps({"items": items}), encoding="utf-8")

    # ---- run pose_rerank.py exactly like the notebook (subprocess, PYTHONPATH=src) ----
    out = tmp_path / "answer_pose.txt"
    env = dict(os.environ, PYTHONPATH=str(SRC))
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pose_rerank.py"),
         "--best", str(ckpt), "--base-ckpt", str(ckpt),
         "--config", str(ROOT / "configs" / "star_v3_10k_kaggle.yaml"),
         "--answer", str(tmp_path / "answer.txt"),
         "--query-json", str(tmp_path / "query_text.json"),
         "--query-index", str(tmp_path / "query_index.txt"),
         "--gt", str(tmp_path / "ground_truth.txt"),
         "--image-root", str(img_root), "--pose-json", str(tmp_path / "pose.json"),
         "--out", str(out), "--topk", "4"],
        env=env, capture_output=True, text=True)

    assert r.returncode == 0, f"pose_rerank crashed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "pose branch=ON" in r.stdout, f"pose branch not loaded:\n{r.stdout}"
    assert "pose-ON cosine" in r.stdout
    # output: one ranked line per query, each a permutation of the 4 ids
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(ids)
    for l in lines:
        assert sorted(l.split()) == sorted(ids)
    metrics = json.loads((out.parent / "pose_metrics.json").read_text())
    assert {"before", "after"} <= set(metrics) and "mAP" in metrics["before"]
