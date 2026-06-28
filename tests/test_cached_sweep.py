"""Cached fusion sweep behavior that must not require auxiliary models when disabled."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


def load_sweep_module():
    path = Path(__file__).resolve().parents[1] / "scripts/sweep_cached_rerank.py"
    spec = importlib.util.spec_from_file_location("sweep_cached_rerank", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def minimal_cache():
    return {
        "metadata": {"candidate_hash": "abc", "topk": 3, "has_ground_truth": True},
        "query_image_ids": ["g0", "g1"],
        "candidate_image_ids": [["g0", "g1", "g2"], ["g1", "g0", "g2"]],
        "pe_scores": torch.tensor([[0.9, 0.8, 0.1], [0.9, 0.7, 0.2]]),
    }


def test_ensemble_off_does_not_load_auxiliary_files():
    sweep = load_sweep_module()
    sets = sweep.retrieval_score_sets(
        minimal_cache(), "off", "/does/not/exist/sig.pt", "/does/not/exist/dfn.pt"
    )
    assert [item["name"] for item in sets] == ["pe_only"]
    assert torch.equal(sets[0]["scores"], minimal_cache()["pe_scores"])


def test_ensemble_compare_requires_both_auxiliary_files():
    sweep = load_sweep_module()
    with pytest.raises(ValueError, match="requires SigLIP2 and DFN"):
        sweep.retrieval_score_sets(minimal_cache(), "compare", None, None)


def test_safe_sort_uses_r1_then_map_then_r10_and_safety_floor():
    sweep = load_sweep_module()
    rows = [
        {"name": "unsafe", "R@1": 0.9, "mAP": 0.9, "R@10": 0.95},
        {"name": "map", "R@1": 0.8, "mAP": 0.82, "R@10": 0.99},
        {"name": "r1", "R@1": 0.81, "mAP": 0.80, "R@10": 0.99},
    ]
    ranked = sweep.safe_sort(rows, r10_floor=0.98)
    assert [row["name"] for row in ranked] == ["r1", "map"]
