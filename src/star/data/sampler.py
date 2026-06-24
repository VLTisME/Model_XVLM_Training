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
        self._epoch = 0
        self.groups: dict = defaultdict(list)
        for idx, g in enumerate(group_ids):
            self.groups[g].append(idx)
        self.group_keys = [g for g in self.groups if g is not None]

    def __len__(self) -> int:
        return self.n // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
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

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)


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

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)


class PairMixedBatchSampler(Sampler[list[int]]):
    """Batches with a fixed small number of mined hard pairs plus random fillers.

    `PairBatchSampler` intentionally spends the whole batch on hard pairs. This sampler is
    gentler for the ITC/ITM mix: each batch starts with exactly `hard_pairs` flattened
    `(anchor, partner)` pairs, then fills the remaining slots with random unique rows.
    """

    def __init__(
        self,
        pairs,
        anchor_groups,
        batch_size: int,
        hard_pairs: int = 4,
        num_samples: int | None = None,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if len(pairs) != len(anchor_groups):
            raise ValueError("pairs and anchor_groups must have the same length")
        if hard_pairs < 0:
            raise ValueError("hard_pairs must be >= 0")
        if 2 * hard_pairs > batch_size:
            raise ValueError("batch_size must be at least 2 * hard_pairs")

        self.pairs = list(pairs)
        self.groups = list(anchor_groups)
        self.batch_size = batch_size
        self.hard_pairs = hard_pairs
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        if num_samples is None:
            max_pair_idx = max((max(a, p) for a, p in self.pairs), default=-1)
            num_samples = max_pair_idx + 1
        self.num_samples = int(num_samples)
        if self.hard_pairs and not self.pairs:
            raise ValueError("pair_mixed needs at least one pair when hard_pairs > 0")

    def __len__(self) -> int:
        if self.hard_pairs * 2 == self.batch_size:
            if self.drop_last:
                return len(self.pairs) // self.hard_pairs
            return (len(self.pairs) + self.hard_pairs - 1) // self.hard_pairs
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        def shuffled_deque(values):
            values = list(values)
            rng.shuffle(values)
            return deque(values)

        pair_queue = shuffled_deque(range(len(self.pairs)))
        filler_queue = shuffled_deque(range(self.num_samples))

        def refill_pairs() -> None:
            pair_queue.extend(shuffled_deque(range(len(self.pairs))))

        def refill_fillers() -> None:
            filler_queue.extend(shuffled_deque(range(self.num_samples)))

        total = len(self)
        for _ in range(total):
            batch: list[int] = []
            used: set[int] = set()
            used_groups: set = set()
            chosen_pairs = 0
            pair_attempts = 0
            max_pair_attempts = max(100, len(self.pairs) * 4)

            while chosen_pairs < self.hard_pairs and pair_attempts < max_pair_attempts:
                pair_attempts += 1
                if not pair_queue:
                    refill_pairs()
                pi = pair_queue.popleft()
                a, p = self.pairs[pi]
                if a in used or p in used or a == p:
                    continue

                group = self.groups[pi]
                if group in used_groups and pair_attempts < len(self.pairs):
                    pair_queue.append(pi)
                    continue

                batch.extend((a, p))
                used.add(a)
                used.add(p)
                used_groups.add(group)
                chosen_pairs += 1

            if chosen_pairs != self.hard_pairs:
                raise RuntimeError(
                    f"Could not build a batch with exactly {self.hard_pairs} hard pairs "
                    f"from {len(self.pairs)} available pairs."
                )

            filler_attempts = 0
            max_filler_attempts = max(100, self.num_samples * 4)
            while len(batch) < self.batch_size and filler_attempts < max_filler_attempts:
                filler_attempts += 1
                if not filler_queue:
                    refill_fillers()
                idx = filler_queue.popleft()
                if idx in used:
                    continue
                batch.append(idx)
                used.add(idx)

            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and batch:
                yield batch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)
