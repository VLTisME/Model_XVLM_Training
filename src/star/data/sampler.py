"""Smart batch samplers.

analyze.md §13. Two strategies:

GroupedBatchSampler — a fraction of each batch comes from ONE scene/action group (soft
co-location of hard negatives); the rest is random for diversity.

PairBatchSampler (V3) — each batch is exactly batch_size//2 (anchor, mined-hard-partner)
PAIRS, anchors drawn from DISTINCT videos. Guarantees:
  - every anchor sees its data-team-mined hard negative in-batch at EVERY step
    (the grouped sampler only achieved this ~once per epoch per video);
  - the only same-video item in the batch is the anchor's own partner (different bucket
    => different sequence_id => a true negative), so no batch slots are wasted on
    same-sequence positives and the ITC negative density is maximal.
"""
from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Iterator

from torch.utils.data import Sampler


class GroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        group_ids: list,
        batch_size: int,
        group_fraction: float = 0.5,
        drop_last: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            group_ids:      per-sample group key (e.g., scene id). `None` => ungrouped pool.
            batch_size:     items per batch.
            group_fraction: fraction of each batch drawn from a single group; rest random.
            drop_last:      drop a final short batch.
        """
        self.batch_size = batch_size
        self.group_fraction = group_fraction
        self.drop_last = drop_last
        self.seed = seed
        self.n = len(group_ids)
        self.groups: dict = defaultdict(list)
        for idx, g in enumerate(group_ids):
            self.groups[g].append(idx)
        self.group_keys = [g for g in self.groups if g is not None]

    def __len__(self) -> int:
        return self.n // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed)
        all_idx = list(range(self.n))
        rng.shuffle(all_idx)
        n_group = int(round(self.batch_size * self.group_fraction))
        pool = iter(all_idx)
        produced = 0
        total = len(self)
        while produced < total:
            batch: list[int] = []
            # grouped portion
            if self.group_keys and n_group > 0:
                gk = rng.choice(self.group_keys)
                members = self.groups[gk]
                k = min(n_group, len(members))
                batch.extend(rng.sample(members, k))
            # random fill (allow re-draw from shuffled pool)
            while len(batch) < self.batch_size:
                try:
                    batch.append(next(pool))
                except StopIteration:
                    rng.shuffle(all_idx)
                    pool = iter(all_idx)
                    batch.append(next(pool))
            yield batch[: self.batch_size]
            produced += 1


class PairBatchSampler(Sampler[list[int]]):
    """Batches of (anchor, partner) pairs with video-distinct anchors.

    Args:
        pairs:         list of (anchor_idx, partner_idx) dataset-row indices.
        anchor_groups: per-PAIR group key (video id) — anchors within one batch must come
                       from distinct groups so cross-pair items are clean negatives.
        batch_size:    total rows per batch (must be even); pairs per batch = batch_size//2.
    Yields flattened batches [a1, p1, a2, p2, ...]. Each pair appears at most once per epoch;
    pairs whose group collides inside the forming batch are deferred to a later batch.
    """

    def __init__(self, pairs, anchor_groups, batch_size: int, drop_last: bool = True, seed: int = 0):
        if batch_size % 2 != 0:
            raise ValueError(f"PairBatchSampler needs an even batch_size, got {batch_size}")
        if len(pairs) != len(anchor_groups):
            raise ValueError("pairs and anchor_groups must have the same length")
        self.pairs = list(pairs)
        self.groups = list(anchor_groups)
        self.k = batch_size // 2
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.pairs) // self.k

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        order = list(range(len(self.pairs)))
        rng.shuffle(order)
        queue = deque(order)
        deferred: deque[int] = deque()
        chosen: list[int] = []
        used_groups: set = set()
        produced = 0

        def flush():
            nonlocal chosen, used_groups, produced
            batch = []
            for pi in chosen:
                a, p = self.pairs[pi]
                batch.extend((a, p))
            chosen, used_groups = [], set()
            produced += 1
            return batch

        while queue or deferred:
            # prefer previously-deferred pairs whose group is now free
            took_deferred = False
            for _ in range(len(deferred)):
                pi = deferred.popleft()
                if self.groups[pi] not in used_groups:
                    chosen.append(pi)
                    used_groups.add(self.groups[pi])
                    took_deferred = True
                    break
                deferred.append(pi)
            if not took_deferred:
                if not queue:
                    break  # only colliding deferred pairs remain -> drop (rare tail)
                pi = queue.popleft()
                if self.groups[pi] in used_groups:
                    deferred.append(pi)
                    continue
                chosen.append(pi)
                used_groups.add(self.groups[pi])
            if len(chosen) == self.k:
                yield flush()
        if chosen and not self.drop_last:
            yield flush()
