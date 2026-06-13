"""PairBatchSampler (V3): pair co-location + video-distinct anchors."""
import pandas as pd
import pytest

from star.data import PairBatchSampler


def test_batches_contain_full_pairs_and_distinct_groups():
    # 6 pairs over 4 videos (video 'A' has 3 pairs -> forces deferral)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    groups = ["A", "A", "A", "B", "C", "D"]
    s = PairBatchSampler(pairs, groups, batch_size=4, seed=0)   # 2 pairs/batch
    batches = list(s)
    assert len(batches) == len(s) == 3
    seen_pairs = set()
    for b in batches:
        assert len(b) == 4
        # flattened as [a1, p1, a2, p2] -> reconstruct pairs and check group-distinctness
        bp = [(b[0], b[1]), (b[2], b[3])]
        gs = [groups[pairs.index(p)] for p in bp]
        assert len(set(gs)) == 2, f"same-video anchors in one batch: {gs}"
        seen_pairs.update(bp)
    assert seen_pairs == set(pairs)             # every pair appears exactly once per epoch


def test_epochs_reshuffle():
    pairs = [(i * 2, i * 2 + 1) for i in range(8)]
    groups = list("ABCDEFGH")
    s = PairBatchSampler(pairs, groups, batch_size=4, seed=0)
    e1, e2 = [tuple(b) for b in s], [tuple(b) for b in s]
    assert e1 != e2                              # different epoch -> different order


def test_odd_batch_size_rejected():
    with pytest.raises(ValueError):
        PairBatchSampler([(0, 1)], ["A"], batch_size=5)


def test_dataset_pairs_mapping():
    # build the dataset object directly (no parquet round-trip: pyarrow-after-torch is a
    # known flaky native crash on Windows; pairs() only reads self.df anyway)
    from star.data import PABDataset
    df = pd.DataFrame([
        # anchors (pair_image_id -> partner), partners, and one row without pair
        dict(image_path="x0.webp", caption="a", split="train", sequence_id="v1_goal",
             scene="v1", video_id=1, image_id="i0", pair_image_id="i1"),
        dict(image_path="x1.webp", caption="b", split="train", sequence_id="v1_went",
             scene="v1", video_id=1, image_id="i1", pair_image_id=None),
        dict(image_path="x2.webp", caption="c", split="train", sequence_id="v2_goal",
             scene="v2", video_id=2, image_id="i2", pair_image_id="i3"),
        dict(image_path="x3.webp", caption="d", split="train", sequence_id="v2_went",
             scene="v2", video_id=2, image_id="i3", pair_image_id=None),
        dict(image_path="x4.webp", caption="e", split="train", sequence_id="v3_goal",
             scene="v3", video_id=3, image_id="i4", pair_image_id="MISSING"),  # partner absent
    ])
    ds = PABDataset.__new__(PABDataset)
    ds.df = df.reset_index(drop=True)
    pairs, groups = ds.pairs()
    assert pairs == [(0, 1), (2, 3)]             # missing-partner anchor skipped
    assert groups == [1, 2]                      # grouped by video_id
