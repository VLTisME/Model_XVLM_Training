"""pose_rerank pure logic: top-K re-rank by pose score + RRF blend + metrics (no model needed)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "pose_rerank", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "pose_rerank.py")
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)


def test_rerank_topk_promotes_gt_pure():
    # GT 'b' is rank-2; pose score prefers it -> pure pose puts it rank-1
    full = ["a", "b", "c", "d"]
    scores = {"a": 0.2, "b": 0.9}                 # only top-2 scored (k=2)
    out = pr.rerank_topk(full, scores, k=2, blend=False)
    assert out == ["b", "a", "c", "d"]


def test_rerank_topk_pure_flips_blend_conservative():
    full = ["a", "b", "c", "d", "e"]
    scores = {"a": 0.1, "b": 0.9}
    pure = pr.rerank_topk(full, scores, k=2, blend=False)
    assert pure == ["b", "a", "c", "d", "e"]      # pure pose flips rank-1<->rank-2 (the goal)
    blend = pr.rerank_topk(full, scores, k=2, blend=True)
    assert blend[2:] == ["c", "d", "e"] and set(blend[:2]) == {"a", "b"}   # tail kept; blend = symmetric tie


def test_metrics_before_after():
    B = pr.metrics([2, 1, 0])                      # ranks 2,1,not-found
    assert abs(B["mAP"] - (0.5 + 1.0) / 3) < 1e-9 and abs(B["R1"] - 0.5) < 1e-9
    A = pr.metrics([1, 1, 0])
    assert A["mAP"] > B["mAP"] and A["R1"] > B["R1"]
