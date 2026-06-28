"""Inference pipeline math: stage-1 ranks, rerank rank bookkeeping, Gale-Shapley, pairwise/RRF."""
import torch

from star.inference import (apply_gale_shapley, gale_shapley_match, pairwise_rerank,
                            ranks_after_rerank, report_from_ranks, rrf_fuse, stage1_ranks)


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


def test_pipeline_accepts_precomputed_candidate_payload(tmp_path):
    """An external RRF block aligns by IDs and bypasses Stage-1 Top-K generation."""
    import numpy as np
    import pandas as pd
    from PIL import Image
    from star.config import Config
    from star.data import PABDataset
    from star.inference import run_pipeline
    from star.models import STARModel

    rng = np.random.default_rng(11)
    rows = []
    for i in range(12):
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).save(
            tmp_path / f"g{i}.jpg"
        )
        rows.append(dict(
            image_path=f"g{i}.jpg", caption=f"query {i}" if i < 4 else "",
            split="valb", sequence_id=f"s{i}", scene=f"s{i}", action="x",
            image_id=f"img{i}", bbox=None, keypoints=None,
        ))
    manifest = tmp_path / "candidate.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)

    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.embed_dim = 32
    model = STARModel(cfg)
    ds = PABDataset(
        str(manifest), str(tmp_path), model.backbone.tokenizer, split="valb", train=False
    )
    candidate_ids = [
        [f"img{q}", "img4", "img5", "img6", "img7", "img8", "img9", "img10", "img11", f"img{(q + 1) % 4}"]
        for q in range(4)
    ]
    payload = {
        "query_image_ids": [f"img{i}" for i in range(4)],
        "candidate_image_ids": candidate_ids,
        "candidate_scores": torch.linspace(1.0, 0.0, 10).repeat(4, 1),
        "pe_raw_ranks": torch.ones(4, dtype=torch.long),
        "pe_selected_ranks": torch.ones(4, dtype=torch.long),
        "stage1_ranks": torch.ones(4, dtype=torch.long),
        "metadata": {"mode": "raw"},
    }
    res = run_pipeline(
        model, ds, "cpu", topk=10, batch_size=4, num_workers=0,
        candidate_payload=payload, stage1_weight=0.25,
    )

    assert res["topk"] == 10
    assert res["num_queries"] == 4 and res["gallery_size"] == 12
    assert res["stage1_raw"]["R@1"] == 1.0
    assert res["pe_selected"]["R@1"] == 1.0
    assert all(len(row) == 10 for row in res["top10"])


def test_cached_rerank_reproduces_pipeline_and_reuses_valid_cache(tmp_path):
    """Cached ITM logits reproduce the legacy additive path without another model pass."""
    import numpy as np
    import pandas as pd
    from PIL import Image
    from star.config import Config
    from star.data import PABDataset
    from star.inference import (evaluate_cached_rerank, load_valid_rerank_cache,
                                prepare_rerank_cache, run_pipeline)
    from star.models import STARModel

    rng = np.random.default_rng(19)
    rows = []
    for i in range(12):
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).save(
            tmp_path / f"g{i}.jpg"
        )
        rows.append(dict(
            image_path=f"g{i}.jpg", caption=f"query {i}" if i < 4 else "",
            split="valb", sequence_id=f"s{i}", scene=f"s{i}", action="x",
            image_id=f"img{i}", bbox=None, keypoints=None,
        ))
    manifest = tmp_path / "cached.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)

    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.embed_dim = 32
    model = STARModel(cfg)
    ds = PABDataset(
        str(manifest), str(tmp_path), model.backbone.tokenizer,
        split="valb", train=False,
    )
    candidate_ids = [
        [f"img{q}", "img4", "img5", "img6", "img7", "img8", "img9", "img10", "img11", f"img{(q + 1) % 4}"]
        for q in range(4)
    ]
    scores = torch.linspace(1.0, 0.0, 10).repeat(4, 1)
    payload = {
        "query_image_ids": [f"img{i}" for i in range(4)],
        "candidate_image_ids": candidate_ids,
        "candidate_scores": scores,
        "candidate_hash": "cache-test-candidates",
        "pe_raw_ranks": torch.ones(4, dtype=torch.long),
        "stage1_ranks": torch.ones(4, dtype=torch.long),
        "metadata": {"mode": "raw"},
    }
    fingerprint = {"manifest": "semantic-19", "checkpoint": "dummy-32"}
    cache_path = tmp_path / "itm_cache.pt"
    cache, reused = prepare_rerank_cache(
        model, ds, "cpu", payload, cache_path, topk=10,
        batch_size=4, num_workers=0, cache_fingerprint=fingerprint,
    )
    assert not reused and cache_path.is_file()

    legacy = run_pipeline(
        model, ds, "cpu", topk=10, batch_size=4, num_workers=0,
        candidate_payload=payload, stage1_weight=0.25,
    )
    cached = evaluate_cached_rerank(
        cache, scores, fusion_family="legacy", fusion_weight=0.25,
        postprocesses=("rerank", "greedy_sca", "gale_shapley"),
    )
    for stage in ("rerank", "greedy_sca", "gale_shapley"):
        assert cached[stage] == legacy[stage]
        assert cached["top10_by_stage"][stage] == legacy["top10_by_stage"][stage]

    loaded, reused = prepare_rerank_cache(
        None, None, "cpu", payload, cache_path, topk=10,
        cache_fingerprint=fingerprint,
    )
    assert reused and torch.equal(loaded["itm_logits"], cache["itm_logits"])
    assert load_valid_rerank_cache(
        cache_path, payload, 10, {**fingerprint, "checkpoint": "changed"}
    ) is None


def test_cached_rerank_fusion_families_are_cpu_only_and_finite():
    from star.inference import evaluate_cached_rerank

    cache = {
        "metadata": {"has_ground_truth": True},
        "gallery_ids": ["g0", "g1", "g2"],
        "query_image_ids": ["g0", "g1"],
        "candidate_image_ids": [["g0", "g1", "g2"], ["g1", "g0", "g2"]],
        "candidate_indices": torch.tensor([[0, 1, 2], [1, 0, 2]]),
        "pe_scores": torch.tensor([[0.9, 0.8, 0.1], [0.9, 0.7, 0.2]]),
        "itm_logits": torch.tensor([[0.2, 0.8, -0.1], [0.1, 0.4, -0.2]]),
        "query_feats": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "gt_pos": torch.tensor([0, 1]),
        "fallback_ranks": torch.ones(2, dtype=torch.long),
    }
    for family in ("legacy", "calibrated", "rank"):
        result = evaluate_cached_rerank(
            cache, cache["pe_scores"], fusion_family=family,
            fusion_weight=1.0, postprocesses=("rerank",),
        )
        assert result["orders"]["rerank"].device.type == "cpu"
        assert torch.isfinite(result["scores"]["rerank"]).all()


def _make_dummy_eval(tmp_path, captions, seed=0):
    """Write a tiny eval manifest + jpgs; `captions[i]` ("" => distractor) drives is_query."""
    import numpy as np
    import pandas as pd
    from PIL import Image

    rng = np.random.default_rng(seed)
    rows = []
    for i, cap in enumerate(captions):
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).save(tmp_path / f"g{i}.jpg")
        rows.append(dict(image_path=f"g{i}.jpg", caption=cap, split="valb",
                         sequence_id=f"s{i}", scene=f"s{i}", action="x",
                         image_id=f"img{i}", bbox=None, keypoints=None))
    m = tmp_path / "m.parquet"
    pd.DataFrame(rows).to_parquet(m, index=False)
    return str(m)


def test_pairwise_rerank_promotes_strongest():
    """Round-robin with a stub comparator head(a,b)=a[:,0]-b[:,0]: the candidate with the largest
    feature-0 beats the most others -> lands at rank 1."""
    feats = torch.tensor([[0.1], [0.9], [0.5], [0.2]])          # [N=4, H=1]
    order = pairwise_rerank(lambda a, b: (a[:, 0] - b[:, 0]), feats)
    assert order[0].item() == 1                                 # idx 1 (0.9) wins the tournament
    assert order.tolist() == [1, 2, 3, 0]


def test_rrf_fuse_combines_rankings():
    # item 2 is high in both lists -> should top the fusion; reciprocal-rank weighted
    fused = rrf_fuse([[2, 0, 1, 3], [2, 1, 0, 3]])
    assert fused[0] == 2 and set(fused) == {0, 1, 2, 3}


def test_run_pipeline_with_pairwise_head(tmp_path):
    """End-to-end: passing a PairwiseHead adds a 'pairwise' stage with valid metrics (dummy model)."""
    from star.config import Config
    from star.data import PABDataset
    from star.inference import run_pipeline
    from star.models import PairwiseHead, STARModel

    m = _make_dummy_eval(tmp_path, [f"person {i}" for i in range(10)], seed=7)
    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.embed_dim = 32
    model = STARModel(cfg)
    ds = PABDataset(m, str(tmp_path), model.backbone.tokenizer, split="valb", train=False)
    # cross-feature dim of this backbone -> build a matching head
    b = ds[0]
    te, _ = model.backbone.encode_text(b["input_ids"].unsqueeze(0), b["attention_mask"].unsqueeze(0))
    ie, _ = model.backbone.encode_image(b["image"].unsqueeze(0))
    dim = model.backbone.cross_feature(ie, te, b["attention_mask"].unsqueeze(0)).size(-1)
    head = PairwiseHead(dim)

    res = run_pipeline(model, ds, "cpu", topk=5, batch_size=4, num_workers=0,
                       pairwise_head=head, pairwise_topn=5)
    assert "pairwise" in res and "rerank" in res
    for k in ("mAP", "R@1", "R@5", "R@10"):
        assert 0.0 <= res["pairwise"][k] <= 1.0
    assert res["num_queries"] == 10 and res["gallery_size"] == 10


def test_pipeline_full_self_gallery_no_distractors(tmp_path):
    """V5 / paper protocol: gallery = the whole test set, EVERY image is both a query and a
    gallery candidate (no distractors). num_queries must equal gallery_size, GT = own image."""
    from star.config import Config
    from star.data import PABDataset
    from star.inference import run_pipeline
    from star.models import STARModel

    n = 12
    m = _make_dummy_eval(tmp_path, [f"person number {i} doing a thing" for i in range(n)], seed=1)
    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.embed_dim = 32
    model = STARModel(cfg)
    ds = PABDataset(m, str(tmp_path), model.backbone.tokenizer, split="valb", train=False)
    res = run_pipeline(model, ds, "cpu", topk=5, batch_size=4, num_workers=0)

    assert res["num_queries"] == n and res["gallery_size"] == n   # every row is query AND gallery
    for stage in ("stage1", "rerank", "gale_shapley"):
        for k in ("mAP", "R@1", "R@5", "R@10"):
            assert 0.0 <= res[stage][k] <= 1.0
    # single GT per query => mAP == MRR exactly, at every stage
    for stage in ("stage1", "rerank", "gale_shapley"):
        assert abs(res[stage]["mAP"] - res[stage]["MRR"]) < 1e-6


def test_pipeline_pose_enabled_checkpoint_without_keypoints(tmp_path):
    """The v3c-on-old-test path: a checkpoint trained with pose_enabled=True must still evaluate
    when the manifest has NO keypoints. The pipeline encodes via backbone.encode_image and never
    calls the pose branch, so it runs (pose-OFF) rather than raising."""
    from star.config import Config
    from star.data import PABDataset
    from star.inference import run_pipeline
    from star.models import STARModel

    m = _make_dummy_eval(tmp_path, ["a person"] * 4 + [""] * 4, seed=2)
    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.embed_dim = 32
    cfg.model.pose_enabled = True                       # build the pose branch...
    model = STARModel(cfg)
    assert model.pose is not None                       # ...it exists in the model
    ds = PABDataset(m, str(tmp_path), model.backbone.tokenizer, split="valb", train=False)
    res = run_pipeline(model, ds, "cpu", topk=4, batch_size=4, num_workers=0)  # must NOT raise

    assert res["num_queries"] == 4 and res["gallery_size"] == 8


def test_pipeline_fuses_pose_when_keypoints_present(tmp_path):
    """With a pose branch AND keypoints in the manifest, the pipeline fuses pose into the image
    feature (matching a pose-ON checkpoint), so the gallery features differ from the pose-OFF run.
    This is the 'eval v3c WITH ViTPose keypoints' path."""
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from star.config import Config
    from star.data import PABDataset
    from star.inference import encode_eval_set
    from star.models import STARModel

    rng = np.random.default_rng(3)
    rows_no, rows_yes = [], []
    for i in range(6):
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).save(tmp_path / f"g{i}.jpg")
        base = dict(image_path=f"g{i}.jpg", caption="a person", split="valb",
                    sequence_id=f"s{i}", scene=f"s{i}", action="x", image_id=f"img{i}", bbox=None)
        rows_no.append({**base, "keypoints": None})
        rows_yes.append({**base, "keypoints": list(rng.random(51).astype(float))})
    p_no, p_yes = tmp_path / "no.parquet", tmp_path / "yes.parquet"
    pd.DataFrame(rows_no).to_parquet(p_no, index=False)
    pd.DataFrame(rows_yes).to_parquet(p_yes, index=False)

    cfg = Config()
    cfg.model.backbone = "dummy"
    cfg.model.embed_dim = 32
    cfg.model.pose_enabled = True
    model = STARModel(cfg)
    tok = model.backbone.tokenizer
    enc_no = encode_eval_set(model, PABDataset(str(p_no), str(tmp_path), tok, split="valb", train=False),
                             "cpu", batch_size=6, num_workers=0)
    enc_yes = encode_eval_set(model, PABDataset(str(p_yes), str(tmp_path), tok, split="valb", train=False),
                              "cpu", batch_size=6, num_workers=0)
    # pose fusion changed the GLOBAL image features...
    assert not torch.allclose(enc_no["gallery_feats"], enc_yes["gallery_feats"])
    # ...but NOT the region embeds the cross-encoder uses (ITM never sees pose)
    assert torch.allclose(enc_no["gallery_embeds"].float(), enc_yes["gallery_embeds"].float())
