from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.data import BBoxAwareTransform  # noqa: E402
from star.pe import PEManifestDataset, PEVisionRetriever  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Extract PE stage-1 features for STAR inference.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--text-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="valb")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="replicate PE over all visible GPUs and split each image batch",
    )
    parser.add_argument("--query-bank-output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = PEManifestDataset(
        args.manifest, args.image_root, args.text_cache, split=args.split
    )
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_id = checkpoint.get("args", {}).get(
        "model", "hf-hub:timm/PE-Core-bigG-14-448"
    )
    model = PEVisionRetriever(model_id).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder = model
    if args.data_parallel and torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(model)
        print(f"PE DataParallel GPUs: {list(range(torch.cuda.device_count()))}")
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"PE image autocast dtype: {amp_dtype}")
    transform = BBoxAwareTransform(
        size=448, enabled=False, mean=model.image_mean, std=model.image_std
    )

    def collate(batch):
        return {
            "image": torch.stack(
                [transform.apply(item["image"], item["bbox"]) for item in batch]
            ),
            "text": torch.stack([item["text_feature"] for item in batch]),
            "image_id": [item["image_id"] for item in batch],
        }

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate,
    )

    image_features, text_features, image_ids = [], [], []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="PE features"):
            with torch.autocast("cuda", dtype=amp_dtype):
                image_features.append(
                    encoder(batch["image"].cuda(non_blocking=True)).float().cpu()
                )
            text_features.append(F.normalize(batch["text"].float(), dim=-1))
            image_ids.extend(batch["image_id"])

    image_features = torch.cat(image_features)
    text_features = torch.cat(text_features)
    gallery_ids, gallery_rows, seen = [], [], set()
    for index, image_id in enumerate(image_ids):
        if image_id not in seen:
            seen.add(image_id)
            gallery_ids.append(image_id)
            gallery_rows.append(index)
    query_rows = [
        index
        for index, caption in enumerate(dataset.df["caption"].fillna("").astype(str))
        if caption.strip()
    ]
    payload = {
        "model_id": model_id,
        "gallery_ids": gallery_ids,
        "gallery_feats": image_features[gallery_rows].half(),
        "query_image_ids": [image_ids[index] for index in query_rows],
        "txt_feats": text_features[query_rows].half(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(
        f"output: {output}; gallery={len(gallery_ids):,}; queries={len(query_rows):,}; "
        f"dim={payload['gallery_feats'].shape[1]}"
    )
    if args.query_bank_output:
        bank_path = Path(args.query_bank_output)
        torch.save(
            {
                "model_id": model_id,
                "query_bank": payload["txt_feats"],
                "query_image_ids": payload["query_image_ids"],
            },
            bank_path,
        )
        print(f"query bank: {bank_path}")


if __name__ == "__main__":
    main()
