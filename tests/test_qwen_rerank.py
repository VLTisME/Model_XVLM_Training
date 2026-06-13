"""qwen_rerank pure logic: before/after metrics + rank-1 promotion (no model needed)."""
import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "qwen_rerank", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "qwen_rerank.py")
qr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(qr)


def test_new_order_promotes_pick():
    assert qr.new_order(["a", "b", "c"], "b") == ["b", "a", "c"]
    assert qr.new_order(["a", "b", "c"], "z") == ["a", "b", "c"]   # pick not in list -> unchanged


def test_rerank_metrics_before_after():
    queries = [
        {"full": ["a", "b", "c"], "gt": "b"},     # GT rank 2 before
        {"full": ["d", "e", "f"], "gt": "d"},     # GT rank 1 before
        {"full": ["g", "h", "i"], "gt": "z"},     # GT not in list -> never found
    ]
    picks = {"0": "b", "1": "d", "2": "g"}         # q0 fixed (b->rank1), q1 keeps d, q2 irrelevant
    B, A = qr.rerank_metrics(queries, picks)
    # before: ranks 2, 1, None
    assert abs(B["mAP"] - (0.5 + 1.0 + 0.0) / 3) < 1e-9
    assert abs(B["R1"] - 1 / 2) < 1e-9             # 1 of 2 found-queries at rank 1
    # after: ranks 1, 1, None
    assert abs(A["mAP"] - (1.0 + 1.0 + 0.0) / 3) < 1e-9
    assert abs(A["R1"] - 1.0) < 1e-9
    assert A["mAP"] > B["mAP"] and A["R1"] > B["R1"]   # VLM promoting GT improves both
