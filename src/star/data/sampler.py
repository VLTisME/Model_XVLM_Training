"""Smart (grouped) batch sampler.

analyze.md §13 (Smart sampler). Putting items from the same scene/action
into the same batch makes the in-batch negatives genuinely hard (free hard negatives), forcing
the model to learn fine-grained discrimination. We mix a grouped fraction with random fill so
diversity is preserved (over-grouping biases training).
"""
from __future__ import annotations

import random
from collections import defaultdict
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
