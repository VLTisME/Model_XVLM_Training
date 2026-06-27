from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a train/public PE query bank for DBSN-style normalization."
    )
    parser.add_argument("--input", required=True, help="Subset JSONL or JSON array")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--model", default="hf-hub:timm/PE-Core-bigG-14-448")
    parser.add_argument("--bank-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def caption_entries(rows: list[dict]) -> list[dict]:
    entries = []
    for row_index, row in enumerate(rows):
        for field, fallback in (
            ("caption_enhanced", "caption"),
            ("hard_c_enhanced", "hard_c"),
        ):
            value = row.get(field)
            selected_field = field
            if value is None or value == "":
                value = row.get(fallback)
                selected_field = fallback
            if value is None or value == "":
                continue
            entries.append(
                {
                    "pool_index": len(entries),
                    "row_index": row_index,
                    "image_id": str(row.get("image_id", "")),
                    "hard_i_id": str(row.get("hard_i_id", "")),
                    "field": selected_field,
                    "text": str(value),
                }
            )
    return entries


def main():
    args = parse_args()
    source = Path(args.input)
    rows = load_rows(source)
    entries = caption_entries(rows)
    if args.bank_size <= 0:
        raise ValueError("--bank-size must be positive")
    if len(entries) < args.bank_size:
        raise ValueError(
            f"Caption pool has {len(entries):,} entries, smaller than bank size "
            f"{args.bank_size:,}"
        )

    selected_indices = random.Random(args.seed).sample(range(len(entries)), args.bank_size)
    selected = [entries[index] for index in selected_indices]
    texts = [entry["text"] for entry in selected]

    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit("Install open_clip_torch before building the PE bank") from exc

    device = torch.device(args.device)
    model, _, _ = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)
    # PE training froze the text tower; the 1.88B visual tower is irrelevant here.
    del model.visual
    model = model.to(device).eval()
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        else torch.float16
    )
    features = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="PE query bank"):
            tokens = tokenizer(texts[start : start + args.batch_size]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                encoded = model.encode_text(tokens)
            features.append(F.normalize(encoded.float(), dim=-1).half().cpu())
    bank = torch.cat(features)
    if bank.shape != (args.bank_size, 1280):
        raise ValueError(
            f"Expected PE bank shape {(args.bank_size, 1280)}, got {tuple(bank.shape)}"
        )
    if not torch.isfinite(bank).all():
        raise ValueError("PE query bank contains non-finite values")

    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "query_bank": bank,
        "model_id": args.model,
        "embedding_dim": int(bank.size(1)),
        "bank_size": int(bank.size(0)),
        "seed": args.seed,
        "source_jsonl": str(source),
        "source_entries": selected,
        "created_at": created_at,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)

    manifest = {
        key: value for key, value in payload.items() if key != "query_bank"
    }
    manifest.update(
        {
            "input_rows": len(rows),
            "caption_pool_entries": len(entries),
            "output": str(output),
            "preserved_text_exactly": True,
            "deduplicated": False,
        }
    )
    manifest_output = Path(args.manifest_output)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"source rows: {len(rows):,}")
    print(f"caption pool: {len(entries):,}")
    print(f"bank: {tuple(bank.shape)} {bank.dtype}")
    print(f"output: {output}")
    print(f"manifest: {manifest_output}")


if __name__ == "__main__":
    main()
