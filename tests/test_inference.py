"""Inference pipeline math: stage-1 ranks, rerank rank bookkeeping, Gale-Shapley."""
import torch

from star.inference import (apply_gale_shapley, gale_shapley_match,
                            ranks_after_rerank, report_from_ranks, stage1_ranks)


def test_stage1_ranks_basic_and_ties():
    sim = torch.tensor([[0.9, 0.5, 0.1],
                        [0.5, 0.5, 0.5]])
    ranks = stage1_ranks(sim, torch.tensor([0, 1]))
    assert ranks.tolist() == [1, 3]            # pessimistic on ties


def test_ranks_after_rerank_inside_and_outside_block():
    # gallery of 5; query0: GT=idx 3 inside top-3; query1: GT=idx 4 OUTSIDE top-3
    sim = torch.tensor([[0.9, 0.8, 0.1, 0.7, 0.0],
                        [0.9, 0.8, 0.7, 0.6, 0.5]])
    gt = torch.tensor([3, 4])
    r1 = stage1_ranks(sim, gt)
    assert r1.tolist() == [3, 5]
    topk_sim, topk_idx = sim.topk(3, dim=1)
    # rerank scores: query0 pushes its GT (idx 3) to the top; query1 reshuffles top-3 only
    final = torch.tensor([[0.1, 0.2, 0.9],     # candidates [0,1,3] -> 3 wins
                          [0.3, 0.2, 0.1]])    # candidates [0,1,2] -> same order
    r2, order2 = ranks_after_rerank(sim, topk_idx, final, gt, r1)
    assert r2.tolist() == [1, 5]               # GT promoted to 1; outside-block unchanged
    assert order2[0].tolist() == [3, 1, 0]


def test_gale_shapley_resolves_conflict():
    # both queries prefer image 7; q1 scores higher -> q1 keeps it, q0 falls to image 2
    order = torch.tensor([[7, 2, 5],
                          [7, 5, 2]])
    scores = torch.tensor([[0.8, 0.7, 0.1],
                           [0.9, 0.2, 0.1]])
    matched = gale_shapley_match(order, scores)
    assert matched.tolist() == [2, 7]


def test_apply_gale_shapley_rank_updates():
    # block of 3 candidates; ranks_in are post-rerank ranks
    order = torch.tensor([[4, 9, 6],
                          [5, 8, 3]])
    gt = torch.tensor([9, 8])                  # q0 GT at rank 2; q1 GT at rank 2
    ranks_in = torch.tensor([2, 2])
    # q0 matched its own GT (9): rank -> 1. q1 matched 3 (was BELOW GT): GT pushed 2 -> 3.
    matched = torch.tensor([9, 3])
    ranks, new_order = apply_gale_shapley(order, matched, ranks_in, gt)
    assert ranks.tolist() == [1, 3]
    assert new_order[0].tolist() == [9, 4, 6]
    assert new_order[1].tolist() == [3, 5, 8]


def test_report_from_ranks():
    rep = report_from_ranks(torch.tensor([1, 2, 10]))
    assert abs(rep["mAP"] - (1 + 0.5 + 0.1) / 3) < 1e-6
    assert abs(rep["R@1"] - 1 / 3) < 1e-6 and abs(rep["R@10"] - 1.0) < 1e-6


def test_full_pipeline_smoke_with_dummy_model(tmp_path):
    """End-to-end run_pipeline on the dummy backbone: 6 queries + 14 gallery rows."""
    import pandas as pd
    from star.config import Config
    from star.data import PABDataset
    from star.inference import run_pipeline
    from star.models import STARModel
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for i in range(14):
        p = tmp_path / f"g{i}.jpg"
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).save(p)
        rows.append(dict(image_path=f"g{i}.jpg",
                         caption=f"a person doing action {i}" if i < 6 else "",
                         split="valb", sequence_id=f"s{i}", scene=f"s{i}",
                         action="x", image_id=f"img{i}", bbox=None, keypoints=None))
    m = tmp_path / "m.parquet"
    pd.DataFrame(rows).to_parquet(m, index=False)

    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.embed_dim = 32
    model = STARModel(cfg)
    ds = PABDataset(str(m), str(tmp_path), model.backbone.tokenizer, split="valb", train=False)
    res = run_pipeline(model, ds, "cpu", topk=5, batch_size=4, num_workers=0)

    assert res["num_queries"] == 6 and res["gallery_size"] == 14 and res["topk"] == 5
    for stage in ("stage1", "rerank", "gale_shapley"):
        assert 0.0 <= res[stage]["mAP"] <= 1.0
    assert len(res["top10"]) == 6 and all(len(t) == 10 for t in res["top10"])
    # invariant: no duplicate images within any query's top-10
    # (GS rank-1 uniqueness only holds for MATCHED queries — with tiny K a query may
    #  legitimately exhaust its candidate list and stay unmatched, so don't assert it)
    for t in res["top10"]:
        assert len(set(t)) == len(t)