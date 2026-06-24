from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.pe import load_pe_text_model  # noqa: E402
from star.pe.data import text_cache_key  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute frozen PE caption embeddings.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="hf-hub:timm/PE-Core-bigG-14-448")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists. Use --overwrite to replace it.")

    frame = pd.read_parquet(args.manifest)
    frame = frame.reset_index(drop=True)
    image_ids = frame["image_id"].astype(str).tolist()
    captions = frame["caption"].fillna("").astype(str).tolist()
    keys = [text_cache_key(image_id, caption) for image_id, caption in zip(image_ids, captions)]
    keep = []
    seen = set()
    for index, key in enumerate(keys):
        if key not in seen:
            seen.add(key)
            keep.append(index)
    image_ids = [image_ids[index] for index in keep]
    captions = [captions[index] for index in keep]
    keys = [keys[index] for index in keep]
    long_caption_rate = sum(len(text.split()) > 72 for text in captions) / max(len(captions), 1)

    model, tokenizer = load_pe_text_model(args.model)
    model = model.to(args.device).eval()
    amp_dtype = (
        torch.bfloat16
        if "cuda" in args.device
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    print(f"PE text autocast dtype: {amp_dtype}")
    unique_captions = list(dict.fromkeys(captions))
    caption_position = {caption: index for index, caption in enumerate(unique_captions)}
    unique_features = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(unique_captions), args.batch_size),
            desc="PE unique text",
        ):
            texts = unique_captions[start : start + args.batch_size]
            tokens = tokenizer(texts).to(args.device)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled="cuda" in args.device,
            ):
                encoded = model.encode_text(tokens, normalize=True)
            unique_features.append(F.normalize(encoded.float(), dim=-1).half().cpu())
    unique_features = torch.cat(unique_features)
    features = torch.stack(
        [unique_features[caption_position[caption]] for caption in captions]
    )

    payload = {
        "model_id": args.model,
        "context_length": 72,
        "image_ids": image_ids,
        "keys": keys,
        "features": features,
        "caption_word_count_gt_72_rate": long_caption_rate,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"output: {output}")
    print(
        f"rows: {len(image_ids):,}; unique captions encoded: {len(unique_captions):,}; "
        f"dim: {payload['features'].shape[1]}"
    )
    print(f"captions with >72 whitespace tokens: {long_caption_rate:.2%}")


if __name__ == "__main__":
    main()
