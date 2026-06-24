from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..data.dataset import _parse_bbox
from ..data.transforms import BBoxAwareTransform


def caption_hash(text: str) -> int:
    digest = hashlib.blake2b(str(text).strip().lower().encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)


def text_cache_key(image_id: str, text: str) -> str:
    return f"{image_id}:{caption_hash(text):016x}"


class PEManifestDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        image_root: str,
        text_cache: str,
        split: str = "train",
    ):
        df = pd.read_parquet(manifest)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.image_root = Path(image_root)
        try:
            payload = torch.load(text_cache, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(text_cache, map_location="cpu")
        if "keys" in payload:
            keys = [str(value) for value in payload["keys"]]
        else:
            keys = [str(value) for value in payload["image_ids"]]
        self.text_by_key = {
            key: payload["features"][index].float()
            for index, key in enumerate(keys)
        }
        required_keys = [
            text_cache_key(str(row["image_id"]), str(row["caption"]))
            if "keys" in payload
            else str(row["image_id"])
            for _, row in self.df.iterrows()
        ]
        missing = [key for key in required_keys if key not in self.text_by_key]
        if missing:
            raise KeyError(f"{len(missing):,} rows are missing cached PE text features. First: {missing[:10]}")
        self.inst_ids = (
            self.df.get("sequence_id", pd.Series(range(len(self.df))))
            .astype("category")
            .cat.codes.values
        )

    def __len__(self):
        return len(self.df)

    def pairs(self) -> tuple[list[tuple[int, int]], list]:
        if "pair_image_id" not in self.df.columns:
            return [], []
        positions = {str(value): i for i, value in enumerate(self.df["image_id"])}
        pairs, groups = [], []
        for i, partner_id in enumerate(self.df["pair_image_id"]):
            if not partner_id or isinstance(partner_id, float):
                continue
            partner = positions.get(str(partner_id))
            if partner is None or partner == i:
                continue
            pairs.append((i, partner))
            groups.append(self.df["video_id"].iat[i] if "video_id" in self.df.columns else i)
        return pairs, groups

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = self.image_root / path
        with Image.open(path) as image:
            image = image.convert("RGB").copy()
        image_id = str(row["image_id"])
        text = str(row["caption"])
        key = text_cache_key(image_id, text)
        if key not in self.text_by_key:
            key = image_id
        return {
            "image": image,
            "text_feature": self.text_by_key[key],
            "instance_id": int(self.inst_ids[index]),
            "caption_hash": caption_hash(text),
            "bbox": _parse_bbox(row.get("bbox")),
            "image_id": image_id,
            "row_index": index,
        }


class PEPairCollator:
    def __init__(self, transform: BBoxAwareTransform, hard_pairs: int | None = None):
        self.transform = transform
        self.hard_pairs = hard_pairs

    def __call__(self, batch: list[dict]) -> dict:
        pair_rows = len(batch) if self.hard_pairs is None else min(len(batch), self.hard_pairs * 2)
        if pair_rows % 2:
            raise ValueError("paired rows must be adjacent and even")
        images = [None] * len(batch)
        partner = torch.full((len(batch),), -1, dtype=torch.long)
        for position in range(0, pair_rows, 2):
            spec = self.transform.sample_spec()
            images[position] = self.transform.apply(
                batch[position]["image"], batch[position]["bbox"], spec
            )
            images[position + 1] = self.transform.apply(
                batch[position + 1]["image"], batch[position + 1]["bbox"], spec
            )
            partner[position], partner[position + 1] = position + 1, position
        for position in range(pair_rows, len(batch)):
            images[position] = self.transform(
                batch[position]["image"], batch[position]["bbox"]
            )
        return {
            "image": torch.stack(images),
            "text_feature": torch.stack([item["text_feature"] for item in batch]),
            "instance_id": torch.tensor([item["instance_id"] for item in batch]),
            "caption_hash": torch.tensor([item["caption_hash"] for item in batch]),
            "partner_index": partner,
            "image_id": [item["image_id"] for item in batch],
        }
