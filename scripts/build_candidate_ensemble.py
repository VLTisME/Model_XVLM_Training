from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.inference import apply_sinkhorn_or_dbsn, stage1_ranks  # noqa: E402


SIGLIP2 = "hf-hub:timm/ViT-gopt-16-SigLIP2-384"
DFN = "hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"


def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def candidate_hash(rows: list[list[str]]) -> str:
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def save_payload(payload: dict, path: str):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"saved: {output}")


def add_common_pe_args(parser):
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--use-pe-dbsn", action="store_true")
    parser.add_argument("--pe-query-bank")
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--ground-truth", action="store_true")


def parse_args():
    parser = argparse.ArgumentParser(description="Build PE-local OpenCLIP RRF candidates.")
    sub = parser.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("pe", help="Select raw or PE-DBSN Top-K candidates")
    add_common_pe_args(pe)

    encode = sub.add_parser("encode", help="Encode selected candidates with one OpenCLIP model")
    encode.add_argument("--manifest", required=True)
    encode.add_argument("--image-root", required=True)
    encode.add_argument("--candidates", required=True)
    encode.add_argument("--model", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--batch-size", type=int, default=8)
    encode.add_argument("--text-batch-size", type=int, default=64)
    encode.add_argument("--workers", type=int, default=4)
    encode.add_argument("--device", default="cuda")
    encode.add_argument("--data-parallel", action="store_true")

    fuse = sub.add_parser("fuse", help="RRF-fuse PE, SigLIP2, and DFN rankings")
    fuse.add_argument("--candidates", required=True)
    fuse.add_argument("--siglip2", required=True)
    fuse.add_argument("--dfn", required=True)
    fuse.add_argument("--output", required=True)
    fuse.add_argument("--rrf-constant", type=int, default=60)
    return parser.parse_args()


def validate_bank(path: str, expected_model: str, expected_dim: int):
    payload = load_payload(path)
    bank = payload.get("query_bank")
    if bank is None:
        raise KeyError(f"query_bank missing from {path}")
    def canonical(value):
        return str(value).removeprefix("hf-hub:")

    if canonical(payload.get("model_id")) != canonical(expected_model):
        raise ValueError(
            f"PE bank model {payload.get('model_id')!r} does not match {expected_model!r}"
        )
    if bank.ndim != 2 or bank.size(1) != expected_dim:
        raise ValueError(f"Expected PE bank [B,{expected_dim}], got {tuple(bank.shape)}")
    if not torch.isfinite(bank).all():
        raise ValueError("PE query bank contains non-finite values")
    norms = bank.float().norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=2e-3, rtol=2e-3):
        raise ValueError("PE query bank is not L2-normalized")
    return payload


def run_pe(args):
    stage = load_payload(args.stage)
    pe_model_id = stage.get("model_id", "hf-hub:timm/PE-Core-bigG-14-448")
    gallery = F.normalize(stage["gallery_feats"].float(), dim=1)
    text = F.normalize(stage["txt_feats"].float(), dim=1)
    if text.size(1) != 1280 or gallery.size(1) != 1280:
        raise ValueError("PE candidate selection expects 1280-dimensional PE features")
    sim_raw = text @ gallery.t()
    sim = sim_raw
    mode = "raw"
    if args.use_pe_dbsn:
        if not args.pe_query_bank:
            raise ValueError("--use-pe-dbsn requires --pe-query-bank")
        validate_bank(args.pe_query_bank, pe_model_id, gallery.size(1))
        sim = apply_sinkhorn_or_dbsn(
            sim_raw,
            gallery,
            query_bank_path=args.pe_query_bank,
            mode="dbsn",
            epsilon=args.epsilon,
            max_iter=args.iterations,
        )
        mode = "pe_dbsn"

    k = min(args.topk, gallery.size(0))
    selected_scores, selected_idx = sim.topk(k, dim=1)
    candidate_ids = [
        [str(stage["gallery_ids"][index]) for index in row]
        for row in selected_idx.tolist()
    ]
    payload = {
        "query_image_ids": [str(value) for value in stage["query_image_ids"]],
        "candidate_image_ids": candidate_ids,
        "candidate_scores": selected_scores.float(),
        "candidate_hash": candidate_hash(candidate_ids),
        "metadata": {
            "mode": mode,
            "topk": k,
            "pe_stage": str(args.stage),
            "pe_model_id": pe_model_id,
            "use_pe_dbsn": bool(args.use_pe_dbsn),
            "epsilon": args.epsilon if args.use_pe_dbsn else None,
            "iterations": args.iterations if args.use_pe_dbsn else None,
            "pe_query_bank": str(args.pe_query_bank) if args.use_pe_dbsn else None,
        },
    }
    if args.ground_truth:
        gallery_pos = {str(value): i for i, value in enumerate(stage["gallery_ids"])}
        try:
            gt_pos = torch.tensor(
                [gallery_pos[str(value)] for value in stage["query_image_ids"]],
                dtype=torch.long,
            )
        except KeyError as exc:
            raise ValueError(f"Ground-truth image is absent from the PE gallery: {exc}") from exc
        payload["pe_raw_ranks"] = stage1_ranks(sim_raw, gt_pos)
        payload["stage1_ranks"] = stage1_ranks(sim, gt_pos)

    unique_count = len({value for row in candidate_ids for value in row})
    payload["metadata"]["unique_candidate_images"] = unique_count
    save_payload(payload, args.output)
    print(f"mode: {mode}")
    print(f"queries: {len(candidate_ids):,}; K: {k}")
    print(f"candidate references: {len(candidate_ids) * k:,}")
    print(f"unique candidate images: {unique_count:,}")


class CandidateImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, paths, preprocess):
        self.image_ids = image_ids
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return tensor, self.image_ids[index]


class ImageEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        return self.model.encode_image(images)


def read_manifest(path: str, image_root: str):
    import pandas as pd

    frame = pd.read_parquet(path)
    captions = frame["caption"].fillna("").astype(str)
    query_rows = frame[captions.str.strip().ne("")]
    gallery_rows = frame.drop_duplicates("image_id", keep="first")
    root = Path(image_root)
    path_by_id = {}
    for row in gallery_rows.itertuples(index=False):
        value = Path(str(row.image_path))
        path_by_id[str(row.image_id)] = value if value.is_absolute() else root / value
    return query_rows, path_by_id


def run_encode(args):
    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit("Install open_clip_torch before candidate encoding") from exc

    candidates = load_payload(args.candidates)
    rows = candidates["candidate_image_ids"]
    unique_ids = list(dict.fromkeys(str(value) for row in rows for value in row))
    query_rows, path_by_id = read_manifest(args.manifest, args.image_root)
    missing = [value for value in unique_ids if value not in path_by_id or not path_by_id[value].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing):,} candidate images; first: {missing[:5]}")
    texts = query_rows["caption"].astype(str).tolist()
    query_image_ids = query_rows["image_id"].astype(str).tolist()
    if query_image_ids != [str(value) for value in candidates["query_image_ids"]]:
        raise ValueError("Manifest query order does not match the PE candidate payload")

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.eval()
    if device.type == "cuda":
        model = model.half()
    model = model.to(device)
    amp_dtype = torch.float16

    text_features = []
    with torch.inference_mode():
        from tqdm.auto import tqdm

        for start in tqdm(range(0, len(texts), args.text_batch_size), desc="text features"):
            tokens = tokenizer(texts[start : start + args.text_batch_size]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                encoded = model.encode_text(tokens)
            text_features.append(F.normalize(encoded.float(), dim=-1).half().cpu())

    encoder = ImageEncoder(model)
    if args.data_parallel and torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        print(f"OpenCLIP DataParallel GPUs: {list(range(torch.cuda.device_count()))}")
    dataset = CandidateImageDataset(
        unique_ids, [path_by_id[value] for value in unique_ids], preprocess
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    image_features, encoded_ids = [], []
    with torch.inference_mode():
        from tqdm.auto import tqdm

        for images, ids in tqdm(loader, desc="candidate image features"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda",
            ):
                encoded = encoder(images)
            image_features.append(F.normalize(encoded.float(), dim=-1).half().cpu())
            encoded_ids.extend(str(value) for value in ids)

    payload = {
        "model_id": args.model,
        "candidate_hash": candidates["candidate_hash"],
        "query_image_ids": query_image_ids,
        "txt_feats": torch.cat(text_features),
        "gallery_ids": encoded_ids,
        "gallery_feats": torch.cat(image_features),
    }
    save_payload(payload, args.output)
    print(f"model: {args.model}")
    print(f"queries: {len(texts):,}")
    print(f"unique candidate images: {len(encoded_ids):,}")
    print(f"feature dimension: {payload['gallery_feats'].size(1)}")


def rank_positions(scores: torch.Tensor) -> torch.Tensor:
    order = scores.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    values = torch.arange(1, scores.size(1) + 1).expand_as(order)
    ranks.scatter_(1, order, values)
    return ranks


def feature_scores(features: dict, candidates: dict) -> torch.Tensor:
    if features.get("candidate_hash") != candidates.get("candidate_hash"):
        raise ValueError(f"Candidate hash mismatch for {features.get('model_id')}")
    if [str(value) for value in features["query_image_ids"]] != [
        str(value) for value in candidates["query_image_ids"]
    ]:
        raise ValueError(f"Query order mismatch for {features.get('model_id')}")
    gallery_pos = {str(value): index for index, value in enumerate(features["gallery_ids"])}
    rows = []
    for query_index, candidate_ids in enumerate(candidates["candidate_image_ids"]):
        try:
            indices = [gallery_pos[str(value)] for value in candidate_ids]
        except KeyError as exc:
            raise ValueError(f"Missing candidate feature for {exc}") from exc
        image = features["gallery_feats"][indices].float()
        text = features["txt_feats"][query_index].float()
        rows.append(image @ text)
    return torch.stack(rows)


def run_fuse(args):
    candidates = load_payload(args.candidates)
    siglip = load_payload(args.siglip2)
    dfn = load_payload(args.dfn)
    sig_scores = feature_scores(siglip, candidates)
    dfn_scores = feature_scores(dfn, candidates)
    pe_scores = torch.as_tensor(candidates["candidate_scores"]).float()

    pe_ranks = rank_positions(pe_scores)
    sig_ranks = rank_positions(sig_scores)
    dfn_ranks = rank_positions(dfn_scores)
    constant = float(args.rrf_constant)
    rrf = (
        1.0 / (constant + pe_ranks.float())
        + 1.0 / (constant + sig_ranks.float())
        + 1.0 / (constant + dfn_ranks.float())
    )
    order = rrf.argsort(dim=1, descending=True)
    candidate_rows = candidates["candidate_image_ids"]
    fused_ids = [
        [candidate_rows[q][index] for index in order[q].tolist()]
        for q in range(len(candidate_rows))
    ]
    fused_scores = torch.gather(rrf, 1, order)
    low = fused_scores.min(dim=1, keepdim=True).values
    high = fused_scores.max(dim=1, keepdim=True).values
    normalized = (fused_scores - low) / (high - low).clamp_min(1e-12)

    payload = {
        "query_image_ids": candidates["query_image_ids"],
        "candidate_image_ids": fused_ids,
        "candidate_scores": normalized,
        "rrf_scores": fused_scores,
        "candidate_hash": candidate_hash(fused_ids),
        "model_ranks": {
            "pe": torch.gather(pe_ranks, 1, order),
            "siglip2": torch.gather(sig_ranks, 1, order),
            "dfn": torch.gather(dfn_ranks, 1, order),
        },
        "pe_raw_ranks": candidates.get("pe_raw_ranks"),
        "pe_selected_ranks": candidates.get("stage1_ranks"),
        "metadata": {
            **candidates.get("metadata", {}),
            "rrf_constant": args.rrf_constant,
            "rrf_weights": {"pe": 1.0, "siglip2": 1.0, "dfn": 1.0},
            "siglip2_model_id": siglip.get("model_id"),
            "dfn_model_id": dfn.get("model_id"),
        },
    }
    if candidates.get("stage1_ranks") is not None:
        fallback = torch.as_tensor(candidates["stage1_ranks"]).long()
        gallery_gt = [str(value) for value in candidates["query_image_ids"]]
        fused_ranks = fallback.clone()
        for query_index, (gt, row) in enumerate(zip(gallery_gt, fused_ids)):
            if gt in row:
                fused_ranks[query_index] = row.index(gt) + 1
        payload["stage1_ranks"] = fused_ranks
    save_payload(payload, args.output)
    print(f"mode: {payload['metadata'].get('mode')}")
    print(f"queries: {len(fused_ids):,}; K: {len(fused_ids[0])}")
    print(f"candidate hash: {payload['candidate_hash']}")


def main():
    args = parse_args()
    if args.command == "pe":
        run_pe(args)
    elif args.command == "encode":
        run_encode(args)
    else:
        run_fuse(args)


if __name__ == "__main__":
    main()
