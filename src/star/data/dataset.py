"""PAB dataset + collate.

Reads the parquet manifest delivered by the DATA TEAM (see README.md "Data contract").
Each item yields: image (tensor), tokenized caption, instance id (for hard-neg/dup masking),
optional bbox (LHP) and keypoints (pose). The tokenizer is injected (from the backbone wrapper)
to keep this file backbone-agnostic.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import LHPTransform, build_eval_transform


def _parse_list(v, expected_len):
    """Parse a manifest cell into a float list of `expected_len`, else None."""
    if v is None or isinstance(v, float):   # NaN / null
        return None
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
    try:
        if v is None or len(v) != expected_len:
            return None
        return [float(x) for x in v]
    except TypeError:
        return None


def _parse_bbox(v):
    return _parse_list(v, 4)


def _parse_kpts(v):
    return _parse_list(v, 17 * 3)


class PABDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        image_root: str,
        tokenizer,
        split: str = "train",
        image_size: int = 384,
        max_token: int = 100,
        train: bool = True,
        lhp_kwargs: dict | None = None,
    ):
        df = pd.read_parquet(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.image_root = Path(image_root)
        self.tokenizer = tokenizer
        self.max_token = max_token
        self.train = train
        if train:
            self.transform = LHPTransform(size=image_size, **(lhp_kwargs or {}))
        else:
            self.transform = build_eval_transform(image_size)
        # stable integer instance id per sequence (so ITC/hard-neg can mask same instance)
        self.inst_ids = self.df.get("sequence_id", pd.Series(range(len(self.df)))).astype("category").cat.codes.values

    def __len__(self) -> int:
        return len(self.df)

    def group_ids(self, key: str = "scene") -> list:
        if key in ("none", None) or key not in self.df.columns:
            return [None] * len(self.df)
        return self.df[key].tolist()

    def pairs(self) -> tuple[list, list]:
        """(anchor_idx, partner_idx) pairs for PairBatchSampler + per-pair group (video).

        Requires manifest columns `image_id` and `pair_image_id` (anchor rows carry the
        data-team-mined hard image's id; non-anchor rows have null). Partners outside this
        split are skipped (cannot happen when the split is by video).
        """
        if "pair_image_id" not in self.df.columns or "image_id" not in self.df.columns:
            return [], []
        pos = {str(iid): i for i, iid in enumerate(self.df["image_id"])}
        group_col = ("video_id" if "video_id" in self.df.columns
                     else "scene" if "scene" in self.df.columns else None)
        pairs, groups = [], []
        for i, pid in enumerate(self.df["pair_image_id"]):
            if pid is None or (isinstance(pid, float)):
                continue
            j = pos.get(str(pid))
            if j is None or j == i:
                continue
            pairs.append((i, j))
            groups.append(self.df[group_col].iat[i] if group_col else i)
        return pairs, groups

    def _load_image(self, rel_path: str) -> Image.Image:
        p = Path(rel_path)
        if not p.is_absolute():
            p = self.image_root / rel_path
        return Image.open(p)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        img = self._load_image(row["image_path"])
        bbox = _parse_bbox(row.get("bbox"))
        if self.train and isinstance(self.transform, LHPTransform):
            image = self.transform(img, bbox)
        else:
            image = self.transform(img.convert("RGB"))

        caption = str(row["caption"])
        tok = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_token,
            return_tensors="pt",
        )
        item = {
            "image": image,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "instance_id": int(self.inst_ids[i]),
            # eval (review fix #3): gallery is keyed by image_id; a row is a query iff it has a caption.
            # the data team adds distractor rows as image-only (empty caption) -> gallery-only.
            "image_id": str(row.get("image_id", row["image_path"])),
            "is_query": bool(caption.strip()),
        }
        # optional pose keypoints (only if the data team supplied them)
        kpts = _parse_kpts(row.get("keypoints")) if "keypoints" in self.df.columns else None
        if kpts is not None:
            item["keypoints"] = torch.tensor(kpts, dtype=torch.float)
        return item


def collate_fn(batch: list[dict]) -> dict:
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "instance_id": torch.tensor([b["instance_id"] for b in batch], dtype=torch.long),
    }
    # keypoints only batched if every item has them (pose branch requires the full batch)
    if all("keypoints" in b for b in batch):
        out["keypoints"] = torch.stack([b["keypoints"] for b in batch])
    return out
