"""STAR-v3 inference pipeline (per inference2.svg, trimmed to the agreed core):

    encode (GLOBAL images, optional pose fusion, no LHP) -> Stage-1 ITC cosine
        -> (2) Top-K filter -> (3) cross-encoder ITM re-rank
        -> optional SCA / Gale-Shapley rank-1 postprocess -> top-10 / query

The default path remains single-retriever. An optional external candidate payload lets a
PE-local ensemble provide its ranked Top-K block without changing ITM or postprocessing.
GNN/k-reciprocal are not included. Sinkhorn/DBSN is optional during candidate generation.
Metrics are reported at every old-test stage (cosine / +rerank / +SCA / +GS) so each
block's contribution is measured, not assumed. Official submit mode skips GT metrics.

Memory design: gallery region embeddings ([G, Ni, H]) are cached on CPU in fp16
(~4 GB for 13.7K images) and gathered per rerank chunk; everything else is tiny.

Rank bookkeeping (exact, no approximation):
  - GT inside Top-K  -> its new rank = its position in the reranked K-block.
  - GT outside Top-K -> rerank only permutes the K items ABOVE it, so its rank is unchanged
    from stage-1.
  - Gale-Shapley moves each query's matched image to rank 1; items previously above it
    shift down by exactly one.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional in tiny test envs
    tqdm = None


def _progress(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable


# --------------------------------------------------------------------------- encoding
def _collate(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
        "is_query": [b["is_query"] for b in batch],
    }
    # keypoints only batched if EVERY item has them (pose branch needs the full batch) — mirrors
    # the train collate. Used to fuse pose into the image feature for pose-trained checkpoints.
    if all("keypoints" in b for b in batch):
        out["keypoints"] = torch.stack([b["keypoints"] for b in batch])
    return out


@torch.no_grad()
def encode_eval_set(model, dataset, device, batch_size: int = 64, num_workers: int = 2):
    """Single pass over the eval manifest.

    Every row contributes its image to the gallery (deduped by image_id, first occurrence
    wins); rows with a caption are queries. Returns CPU tensors:
        gallery: feats [G, d] fp32 · embeds [G, Ni, H] fp16 · ids list
        queries: txt_feats [Q, d] · txt_embeds [Q, L, H] · masks [Q, L] · gt_pos [Q]

    Pose: if the model has a pose branch AND the batch carries keypoints, pose is fused into the
    GLOBAL image feature (`img_feat`), matching how a pose-ON checkpoint was trained. The region
    embeds (`img_embeds`, used by the cross-encoder) are left untouched — ITM never uses pose.
    A model with no pose branch (or a manifest with no keypoints) is evaluated pose-OFF.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate)
    use_amp = "cuda" in str(device)
    pose = getattr(model, "pose", None)

    id_to_pos: dict = {}
    g_feats, g_embeds, g_ids = [], [], []
    q_tfeats, q_tembeds, q_masks, q_img_ids = [], [], [], []

    for batch in _progress(loader, desc="encode images/text", total=len(loader), leave=False):
        image = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            img_embeds, img_feat = model.backbone.encode_image(image)
        if pose is not None and "keypoints" in batch:
            img_feat = pose(img_feat.float(), batch["keypoints"].to(device).float())
        for r in range(image.size(0)):
            iid = batch["image_id"][r]
            if iid not in id_to_pos:
                id_to_pos[iid] = len(g_ids)
                g_ids.append(iid)
                g_feats.append(img_feat[r].float().cpu())
                g_embeds.append(img_embeds[r].half().cpu())
        q_rows = [r for r, q in enumerate(batch["is_query"]) if q]
        if q_rows:
            ids = batch["input_ids"][q_rows].to(device)
            mask = batch["attention_mask"][q_rows].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                txt_embeds, txt_feat = model.backbone.encode_text(ids, mask)
            q_tfeats.append(txt_feat.float().cpu())
            q_tembeds.append(txt_embeds.float().cpu())
            q_masks.append(mask.cpu())
            q_img_ids.extend(batch["image_id"][r] for r in q_rows)

    gallery_feats = torch.stack(g_feats)                        # [G, d]
    gallery_embeds = torch.stack(g_embeds)                      # [G, Ni, H] fp16
    txt_feats = torch.cat(q_tfeats)                             # [Q, d]
    txt_embeds = torch.cat(q_tembeds)                           # [Q, L, H]
    masks = torch.cat(q_masks)                                  # [Q, L]
    gt_pos = torch.tensor([id_to_pos[i] for i in q_img_ids])    # [Q]
    return dict(gallery_feats=gallery_feats, gallery_embeds=gallery_embeds, gallery_ids=g_ids,
                txt_feats=txt_feats, txt_embeds=txt_embeds, txt_masks=masks, gt_pos=gt_pos,
                query_image_ids=q_img_ids)


def load_stage1_features(payload, enc: dict) -> tuple[Tensor, Tensor]:
    """Load PE features and reorder them to the X-VLM gallery/query order."""
    if payload is None:
        return enc["gallery_feats"], enc["txt_feats"]
    if isinstance(payload, (str, bytes, Path)):
        try:
            payload = torch.load(payload, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(payload, map_location="cpu")
    gallery_pos = {str(value): i for i, value in enumerate(payload["gallery_ids"])}
    query_positions: dict[str, list[int]] = {}
    for i, value in enumerate(payload["query_image_ids"]):
        query_positions.setdefault(str(value), []).append(i)
    query_cursor = {key: 0 for key in query_positions}
    try:
        gallery_order = [gallery_pos[str(value)] for value in enc["gallery_ids"]]
        query_order = []
        for value in enc["query_image_ids"]:
            key = str(value)
            cursor = query_cursor[key]
            query_order.append(query_positions[key][cursor])
            query_cursor[key] += 1
    except (KeyError, IndexError) as exc:
        raise ValueError("PE stage-1 payload does not match the X-VLM manifest order/IDs") from exc
    gallery = payload["gallery_feats"][gallery_order].float()
    text = payload["txt_feats"][query_order].float()
    return gallery, text


def _payload_query_order(payload_ids: list, encoded_ids: list) -> list[int]:
    """Align payload queries by GT image ID while preserving duplicate occurrences."""
    positions: dict[str, list[int]] = {}
    for index, value in enumerate(payload_ids):
        positions.setdefault(str(value), []).append(index)
    cursor = {key: 0 for key in positions}
    order = []
    try:
        for value in encoded_ids:
            key = str(value)
            offset = cursor[key]
            order.append(positions[key][offset])
            cursor[key] += 1
    except (KeyError, IndexError) as exc:
        raise ValueError(
            "Candidate payload does not match the X-VLM manifest query order/IDs"
        ) from exc
    return order


def load_candidate_payload(payload, enc: dict) -> dict | None:
    """Load an externally ranked Top-K block and align it to the encoded eval set.

    The payload stores image IDs rather than relying on either model's gallery order. This
    keeps PE/OpenCLIP candidate generation independent from X-VLM manifest ordering.
    """
    if payload is None:
        return None
    if isinstance(payload, (str, bytes, Path)):
        try:
            payload = torch.load(payload, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(payload, map_location="cpu")

    required = {"query_image_ids", "candidate_image_ids", "candidate_scores"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Candidate payload is missing fields: {sorted(missing)}")

    query_order = _payload_query_order(payload["query_image_ids"], enc["query_image_ids"])
    gallery_pos = {str(value): index for index, value in enumerate(enc["gallery_ids"])}
    candidate_rows = [payload["candidate_image_ids"][index] for index in query_order]
    if not candidate_rows or not candidate_rows[0]:
        raise ValueError("Candidate payload contains no candidates")
    width = len(candidate_rows[0])
    if any(len(row) != width for row in candidate_rows):
        raise ValueError("Candidate payload rows have inconsistent lengths")
    if any(len({str(value) for value in row}) != width for row in candidate_rows):
        raise ValueError("Candidate payload contains duplicate image IDs within a query")
    try:
        indices = torch.tensor(
            [[gallery_pos[str(value)] for value in row] for row in candidate_rows],
            dtype=torch.long,
        )
    except KeyError as exc:
        raise ValueError(f"Candidate image is missing from the X-VLM gallery: {exc}") from exc

    scores = torch.as_tensor(payload["candidate_scores"])[query_order].float()
    if scores.shape != indices.shape:
        raise ValueError(
            f"Candidate score shape {tuple(scores.shape)} does not match "
            f"candidate indices {tuple(indices.shape)}"
        )
    if not torch.isfinite(scores).all():
        raise ValueError("Candidate payload contains non-finite scores")

    result = {
        "indices": indices,
        "scores": scores,
        "metadata": payload.get("metadata", {}),
    }
    for key in ("pe_raw_ranks", "pe_selected_ranks", "stage1_ranks"):
        if key in payload and payload[key] is not None:
            ranks = torch.as_tensor(payload[key])[query_order].long()
            if ranks.numel() != indices.size(0):
                raise ValueError(f"{key} length does not match the query count")
            result[key] = ranks
    return result


# --------------------------------------------------------------------------- stage 1
def stage1_ranks(sim: Tensor, gt_pos: Tensor) -> Tensor:
    """1-based rank of GT under cosine scores (pessimistic on ties, same as star.metrics)."""
    q = torch.arange(sim.size(0))
    gt_score = sim[q, gt_pos].unsqueeze(1)
    greater = (sim > gt_score).sum(dim=1)
    ties = (sim == gt_score).sum(dim=1) - 1
    return greater + ties + 1


# --------------------------------------------------------------------------- optional Sinkhorn / DBSN
def sinkhorn_normalize(sim: Tensor, epsilon: float = 0.05, max_iter: int = 20) -> Tensor:
    """Balanced assignment-style normalization over a query-gallery score matrix.

    Returns normalized scores with the same shape as `sim`. This is used only before Top-K.
    """
    z = sim.float() / max(float(epsilon), 1e-6)
    z = z - z.max()
    p = torch.exp(z).clamp_min(1e-12)
    for _ in range(max_iter):
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        p = p / p.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return p


def apply_sinkhorn_or_dbsn(sim: Tensor, gallery_feats: Tensor, query_bank_path: str | None = None,
                           mode: str = "sinkhorn", epsilon: float = 0.05,
                           max_iter: int = 20) -> Tensor:
    """Apply plain Sinkhorn or DBSN-style normalization.

    DBSN mode stacks an external query bank above the current query matrix while balancing,
    then returns the normalized rows corresponding to the current queries.
    """
    mode = str(mode).lower()
    if mode not in {"sinkhorn", "dbsn"}:
        raise ValueError("mode must be 'sinkhorn' or 'dbsn'")
    if mode == "sinkhorn":
        return sinkhorn_normalize(sim, epsilon=epsilon, max_iter=max_iter)

    if not query_bank_path:
        raise ValueError("DBSN mode requires query_bank_path")
    try:
        payload = torch.load(query_bank_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(query_bank_path, map_location="cpu")
    bank = payload.get("query_bank")
    if bank is None:
        raise KeyError(f"query_bank not found in {query_bank_path}")
    bank = bank.float()
    bank_sim = bank @ gallery_feats.float().t()
    combined = torch.cat([sim.float(), bank_sim], dim=0)
    normalized = sinkhorn_normalize(combined, epsilon=epsilon, max_iter=max_iter)
    return normalized[:sim.size(0)]


# --------------------------------------------------------------------------- (3) rerank
@torch.no_grad()
def _itm_rerank_worker(model, gallery_embeds: Tensor, txt_embeds: Tensor, txt_masks: Tensor,
                       topk_idx: Tensor, q_indices: list[int], device,
                       pair_chunk: int = 50, label: str = "gpu",
                       progress_every: int = 50) -> tuple[list[int], Tensor]:
    """Worker for a query shard. Returns scores in the same order as q_indices."""
    model.eval()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_amp = "cuda" in str(device)
    K = topk_idx.size(1)
    out = torch.empty(len(q_indices), K)
    total_pairs = len(q_indices) * K
    t0 = time.time()
    with torch.no_grad():
        iterator = enumerate(q_indices)
        if len(q_indices) and tqdm is not None and len(q_indices) == topk_idx.size(0):
            iterator = enumerate(_progress(q_indices, desc=f"ITM rerank {total_pairs:,} pairs", leave=True))
        for local_i, qi in iterator:
            t_emb = txt_embeds[qi].to(device)
            t_mask = txt_masks[qi].to(device)
            idx = topk_idx[qi]
            for s in range(0, K, pair_chunk):
                sel = idx[s:s + pair_chunk]
                img = gallery_embeds[sel].to(device).float()             # [c, Ni, H]
                te = t_emb.unsqueeze(0).expand(img.size(0), -1, -1)
                tm = t_mask.unsqueeze(0).expand(img.size(0), -1)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model.backbone.itm_logits(img, te, tm)      # [c, 2]
                out[local_i, s:s + len(sel)] = logits.float()[:, 1].cpu()
            if progress_every and ((local_i + 1) % progress_every == 0 or local_i + 1 == len(q_indices)):
                done_pairs = (local_i + 1) * K
                elapsed = max(time.time() - t0, 1e-6)
                speed = done_pairs / elapsed
                eta = (total_pairs - done_pairs) / max(speed, 1e-6)
                print(f"{label} ITM rerank {local_i + 1:,}/{len(q_indices):,} queries "
                      f"({done_pairs:,}/{total_pairs:,} pairs); "
                      f"{speed:.1f} pairs/s; eta {eta / 60:.1f} min",
                      flush=True)
    return q_indices, out


def _normalize_rerank_models(model, device, rerank_models):
    if not rerank_models:
        return [(model, torch.device(device))]
    specs = []
    for item in rerank_models:
        if isinstance(item, dict):
            m = item["model"]
            d = torch.device(item.get("device", next(m.parameters()).device))
        else:
            m, d = item
            d = torch.device(d)
        specs.append((m, d))
    return specs


@torch.no_grad()
def itm_rerank(model, gallery_embeds: Tensor, txt_embeds: Tensor, txt_masks: Tensor,
               topk_idx: Tensor, device, pair_chunk: int = 50,
               rerank_models=None) -> Tensor:
    """ITM logit[:,1] for each (query, top-K image) pair. Returns [Q, K] fp32 cpu.

    `rerank_models` may be a list of `(model, device)` pairs. When provided, queries are
    sharded across those models, which is the intended 2xT4 Kaggle speedup path.
    """
    specs = _normalize_rerank_models(model, device, rerank_models)
    Q, K = topk_idx.shape
    out = torch.empty(Q, K)
    if len(specs) <= 1:
        q_indices, scores = _itm_rerank_worker(
            specs[0][0], gallery_embeds, txt_embeds, txt_masks, topk_idx,
            list(range(Q)), specs[0][1], pair_chunk=pair_chunk,
            label=str(specs[0][1]), progress_every=25,
        )
        out[q_indices] = scores
        return out

    shards = [list(range(i, Q, len(specs))) for i in range(len(specs))]
    print("ITM multi-GPU rerank:",
          ", ".join(f"{device}:{len(qs):,}q" for (_, device), qs in zip(specs, shards)),
          flush=True)
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        futures = []
        for worker_i, ((m, d), q_indices) in enumerate(zip(specs, shards)):
            if not q_indices:
                continue
            futures.append(pool.submit(
                _itm_rerank_worker,
                m, gallery_embeds, txt_embeds, txt_masks, topk_idx,
                q_indices, d, pair_chunk, f"gpu{worker_i}:{d}", 50,
            ))
        for fut in as_completed(futures):
            q_indices, scores = fut.result()
            out[q_indices] = scores
    return out


def ranks_after_rerank(sim: Tensor | None, topk_idx: Tensor, final_scores: Tensor,
                       gt_pos: Tensor, ranks_s1: Tensor) -> tuple[Tensor, Tensor]:
    """Exact GT ranks after reordering the K-block by final_scores (desc).

    Returns (ranks [Q], order [Q, K] = topk_idx reordered by the new scores).
    """
    order_in_k = final_scores.argsort(dim=1, descending=True)            # [Q, K]
    new_order = torch.gather(topk_idx, 1, order_in_k)                    # gallery idx, reranked
    ranks = ranks_s1.clone()
    for qi in range(topk_idx.size(0)):
        hit = (new_order[qi] == gt_pos[qi]).nonzero(as_tuple=True)[0]
        if len(hit):
            ranks[qi] = int(hit[0]) + 1          # GT inside K-block -> new position
        # else: only the K items above it were permuted -> stage-1 rank unchanged
    return ranks, new_order


def scores_for_order(topk_idx: Tensor, final_scores: Tensor, order: Tensor) -> Tensor:
    """Gather final_scores so they align with an arbitrary top-K order."""
    out = torch.empty_like(order, dtype=final_scores.dtype)
    for qi in range(order.size(0)):
        score_by_gid = {int(g): float(s) for g, s in zip(topk_idx[qi], final_scores[qi])}
        out[qi] = torch.tensor([score_by_gid[int(g)] for g in order[qi]],
                               dtype=final_scores.dtype)
    return out


def ranks_after_order(order: Tensor, gt_pos: Tensor, ranks_fallback: Tensor) -> Tensor:
    """Exact GT ranks after a top-K order. GT outside K keeps the fallback rank."""
    ranks = ranks_fallback.clone()
    for qi in range(order.size(0)):
        hit = (order[qi] == gt_pos[qi]).nonzero(as_tuple=True)[0]
        if len(hit):
            ranks[qi] = int(hit[0]) + 1
    return ranks


def build_top10(order: Tensor, sim: Tensor | None, gallery_ids: list[str]) -> list[list[str]]:
    """Top-10 = chosen K-block order followed by the original ITC tail."""
    if sim is None and order.size(1) < 10:
        raise ValueError("Candidate-only Top-10 requires at least 10 candidates per query")
    s1_order = sim.argsort(dim=1, descending=True) if sim is not None else None
    top10 = []
    for q in range(order.size(0)):
        seen = order[q].tolist()
        seen_set = set(seen)
        tail = (
            [int(g) for g in s1_order[q] if int(g) not in seen_set]
            if s1_order is not None else []
        )
        full = (seen + tail)[:10]
        top10.append([gallery_ids[g] for g in full])
    return top10


def top1_conflict_count(order: Tensor) -> int:
    top1 = order[:, 0]
    return int(top1.numel() - torch.unique(top1).numel())


def greedy_sca(order: Tensor, scores: Tensor, query_feats: Tensor | None = None,
               max_iter: int = 10, text_sim_threshold: float = 0.96,
               swap_gain: float = 0.01) -> tuple[Tensor, Tensor]:
    """Similarity Coverage Analysis style postprocess.

    This is intentionally conservative:
      1) resolve duplicate rank-1 claims by keeping the highest-confidence query;
      2) for very similar queries with crossed top candidates, accept a swap only when
         the pairwise score sum improves. It is an ablation stage, not the default submit
         choice unless it beats Gale-Shapley on old-test.
    """
    Q, K = order.shape
    assigned = order[:, 0].clone()

    for _ in range(max_iter):
        changed = False
        holders: dict[int, list[int]] = {}
        for q, gid in enumerate(assigned.tolist()):
            holders.setdefault(int(gid), []).append(q)
        occupied = set(int(x) for x in assigned.tolist())
        for gid, qs in holders.items():
            if len(qs) <= 1:
                continue
            qs = sorted(qs, key=lambda q: float(scores[q, 0]), reverse=True)
            for q in qs[1:]:
                old = int(assigned[q])
                replacement = None
                for cand in order[q].tolist():
                    cand = int(cand)
                    if cand == old:
                        continue
                    if cand not in occupied:
                        replacement = cand
                        break
                if replacement is None:
                    for cand in order[q].tolist():
                        cand = int(cand)
                        if cand != old:
                            replacement = cand
                            break
                if replacement is not None and replacement != old:
                    assigned[q] = replacement
                    occupied.discard(old)
                    occupied.add(replacement)
                    changed = True
        if not changed:
            break

    if query_feats is not None and Q > 1:
        feats = torch.nn.functional.normalize(query_feats.float(), dim=1)
        qsim = feats @ feats.t()
        score_lookup = []
        pos_lookup = []
        for q in range(Q):
            score_lookup.append({int(g): float(s) for g, s in zip(order[q], scores[q])})
            pos_lookup.append({int(g): i for i, g in enumerate(order[q].tolist())})
        for i in range(Q):
            for j in range(i + 1, Q):
                if float(qsim[i, j]) < text_sim_threshold:
                    continue
                ai, aj = int(assigned[i]), int(assigned[j])
                if ai == aj:
                    continue
                if aj not in score_lookup[i] or ai not in score_lookup[j]:
                    continue
                if pos_lookup[i][aj] > 2 or pos_lookup[j][ai] > 2:
                    continue
                current = score_lookup[i][ai] + score_lookup[j][aj]
                swapped = score_lookup[i][aj] + score_lookup[j][ai]
                if swapped > current + swap_gain:
                    tmp = int(assigned[i])
                    assigned[i] = assigned[j]
                    assigned[j] = tmp

    new_order = order.clone()
    for q in range(Q):
        chosen = int(assigned[q])
        row = order[q].tolist()
        if chosen in row:
            new_order[q] = torch.tensor([chosen] + [x for x in row if int(x) != chosen],
                                        dtype=order.dtype)
    return new_order, assigned


# --------------------------------------------------------------------------- (5) Gale-Shapley
def gale_shapley_match(order: Tensor, scores: Tensor) -> Tensor:
    """Deferred acceptance. Queries propose down their (already reranked) candidate lists;
    an image keeps the highest-scoring proposer. Returns matched gallery idx per query
    (-1 if a query exhausts its list).

    Args:
        order:  [Q, K] gallery indices, each row sorted by that query's preference (desc).
        scores: [Q, K] the corresponding scores (image-side preference = proposer's score).
    """
    Q, K = order.shape
    next_choice = [0] * Q
    holder: dict[int, int] = {}          # gallery idx -> query currently held
    hold_score: dict[int, float] = {}
    free = list(range(Q))
    while free:
        q = free.pop()
        while next_choice[q] < K:
            c = int(order[q, next_choice[q]])
            s = float(scores[q, next_choice[q]])
            next_choice[q] += 1
            if c not in holder:
                holder[c], hold_score[c] = q, s
                break
            if s > hold_score[c]:
                loser = holder[c]
                holder[c], hold_score[c] = q, s
                free.append(loser)
                break
        # ran out of candidates -> stays unmatched
    matched = torch.full((Q,), -1, dtype=torch.long)
    for c, q in holder.items():
        matched[q] = c
    return matched


def apply_gale_shapley(order: Tensor, matched: Tensor, ranks_in: Tensor,
                       gt_pos: Tensor) -> tuple[Tensor, Tensor]:
    """Move each query's matched image to rank 1 (keep ranks 2..K from the rerank order).

    Exact rank update: if GT == matched -> 1; else GT shifts down by one iff the matched
    image was previously BELOW the GT (moving it above pushes GT down); unchanged otherwise.
    """
    Q, K = order.shape
    new_order = order.clone()
    ranks = ranks_in.clone()
    for qi in range(Q):
        m = int(matched[qi])
        if m < 0:
            continue
        row = order[qi].tolist()
        pos_m = row.index(m) + 1                     # 1-based position of matched in the block
        new_order[qi] = torch.tensor([m] + [x for x in row if x != m])
        if int(gt_pos[qi]) == m:
            ranks[qi] = 1                            # GT itself promoted
        elif int(ranks_in[qi]) <= K and pos_m > int(ranks_in[qi]):
            ranks[qi] = ranks_in[qi] + 1             # an item from BELOW GT jumped above it
        # matched already above GT (pos_m < rank) or GT outside the block -> rank unchanged
    return ranks, new_order


# --------------------------------------------------------------------------- (pairwise / duo)
@torch.no_grad()
def pairwise_features(model, gallery_embeds: Tensor, txt_embeds: Tensor, txt_masks: Tensor,
                      idx: Tensor, device) -> Tensor:
    """Cross-encoder fused [CLS] feature for each (query, candidate) in `idx` [Q, N] -> [Q, N, H] cpu.

    N is small (top-N to compare, e.g. 10) so this is ~N cross-encoder forwards per query — cheap.
    """
    model.eval()
    use_amp = "cuda" in str(device)
    Q, N = idx.shape
    out = None
    for qi in range(Q):
        t_emb = txt_embeds[qi].unsqueeze(0).expand(N, -1, -1).to(device)
        t_mask = txt_masks[qi].unsqueeze(0).expand(N, -1).to(device)
        img = gallery_embeds[idx[qi]].to(device).float()                 # [N, Ni, H]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            h = model.backbone.cross_feature(img, t_emb, t_mask)         # [N, H]
        if out is None:
            out = torch.empty(Q, N, h.size(-1))
        out[qi] = h.float().cpu()
    return out


@torch.no_grad()
def pairwise_rerank(head, feats: Tensor) -> Tensor:
    """Round-robin: head(a,b)=logit P(a>b) over N candidates -> Borda order (local perm [N]).

    `head` is any callable mapping ([M,H],[M,H]) -> [M] logits (a PairwiseHead or a test stub).
    """
    N = feats.size(0)
    a = feats.unsqueeze(1).expand(N, N, -1).reshape(N * N, -1)
    b = feats.unsqueeze(0).expand(N, N, -1).reshape(N * N, -1)
    P = torch.sigmoid(head(a, b)).reshape(N, N)
    P.fill_diagonal_(0.0)
    return P.sum(dim=1).argsort(descending=True)                         # who beats the most others


def rrf_fuse(orders: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion of several ranked lists over the SAME item set (Cormack 2009)."""
    score: dict[int, float] = {}
    for order in orders:
        for rank, idx in enumerate(order):
            score[int(idx)] = score.get(int(idx), 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: score[i], reverse=True)


# --------------------------------------------------------------------------- reporting
def report_from_ranks(ranks: Tensor, ks=(1, 5, 10, 50, 200)) -> dict[str, float]:
    r = ranks.float()
    rep = {"mAP": float((1.0 / r).mean()), "MRR": float((1.0 / r).mean())}
    for k in ks:
        rep[f"R@{k}"] = float((r <= k).float().mean())
    return rep


def score_calibration_report(itm: Tensor, stage1: Tensor, final: Tensor,
                             stage1_weight: float) -> dict[str, float | int]:
    """Describe whether the normalized Stage-1 prior materially changes ITM rankings."""
    def stats(prefix: str, values: Tensor) -> dict[str, float]:
        values = values.float()
        return {
            f"{prefix}_min": float(values.min()),
            f"{prefix}_max": float(values.max()),
            f"{prefix}_mean": float(values.mean()),
            f"{prefix}_std": float(values.std()),
        }

    def median_top2_gap(values: Tensor) -> float:
        if values.size(1) < 2:
            return 0.0
        top2 = values.float().topk(2, dim=1).values
        return float((top2[:, 0] - top2[:, 1]).median())

    weighted = float(stage1_weight) * stage1.float()
    report = {
        **stats("itm", itm),
        **stats("stage1", stage1),
        **stats("weighted_stage1", weighted),
        "stage1_weight": float(stage1_weight),
        "itm_median_top2_gap": median_top2_gap(itm),
        "stage1_median_top2_gap": median_top2_gap(stage1),
        "final_median_top2_gap": median_top2_gap(final),
        "top1_changed_by_stage1": int(
            (itm.argmax(dim=1) != final.argmax(dim=1)).sum()
        ),
        "query_count": int(itm.size(0)),
    }
    return report


# --------------------------------------------------------------------------- cached rerank
RERANK_CACHE_VERSION = 2


def _stable_hash(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_torch_payload(payload):
    if isinstance(payload, (str, bytes, Path)):
        try:
            return torch.load(payload, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(payload, map_location="cpu")
    return payload


def _cache_expected_metadata(candidate_payload, topk: int,
                             cache_fingerprint: dict | None = None) -> dict:
    payload = _load_torch_payload(candidate_payload)
    rows = payload.get("candidate_image_ids")
    if not rows:
        raise ValueError("A non-empty candidate payload is required for cached reranking")
    width = len(rows[0])
    if width != int(topk) or any(len(row) != width for row in rows):
        raise ValueError(f"Candidate payload width {width} does not match topk={topk}")
    candidate_digest = payload.get("candidate_hash") or _stable_hash(rows)
    return {
        "cache_version": RERANK_CACHE_VERSION,
        "candidate_hash": str(candidate_digest),
        "query_hash": _stable_hash([str(value) for value in payload["query_image_ids"]]),
        "topk": int(topk),
        "fingerprint": dict(cache_fingerprint or {}),
    }


def load_valid_rerank_cache(path: str | Path, candidate_payload, topk: int,
                            cache_fingerprint: dict | None = None) -> dict | None:
    """Load a reusable CPU rerank cache, returning None when its inputs changed."""
    path = Path(path)
    if not path.is_file():
        return None
    cache = _load_torch_payload(path)
    expected = _cache_expected_metadata(candidate_payload, topk, cache_fingerprint)
    actual = cache.get("metadata", {})
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items() if actual.get(key) != value
    }
    if mismatches:
        print(f"rerank cache invalid: {path}", flush=True)
        for key, values in mismatches.items():
            print(f"  {key}: expected={values['expected']!r} "
                  f"actual={values['actual']!r}", flush=True)
        return None
    required = {
        "gallery_ids", "query_image_ids", "query_labels", "candidate_image_ids", "candidate_indices",
        "pe_scores", "itm_logits", "query_feats",
    }
    missing = required.difference(cache)
    if missing:
        print(f"rerank cache missing fields: {sorted(missing)}", flush=True)
        return None
    if actual.get("has_ground_truth"):
        gt_missing = {"gt_pos", "pe_raw_ranks", "fallback_ranks"}.difference(cache)
        if gt_missing:
            print(f"rerank cache missing GT fields: {sorted(gt_missing)}", flush=True)
            return None
    expected_shape = (len(cache["query_image_ids"]), int(topk))
    for key in ("candidate_indices", "pe_scores", "itm_logits"):
        value = torch.as_tensor(cache[key])
        if tuple(value.shape) != expected_shape:
            print(f"rerank cache {key} shape {tuple(value.shape)} != "
                  f"{expected_shape}", flush=True)
            return None
        if key != "candidate_indices" and not torch.isfinite(value).all():
            print(f"rerank cache {key} contains non-finite values", flush=True)
            return None
    if any(len(row) != int(topk) for row in cache["candidate_image_ids"]):
        print("rerank cache candidate rows have inconsistent width", flush=True)
        return None
    return cache


@torch.no_grad()
def prepare_rerank_cache(model, dataset, device, candidate_payload,
                         cache_path: str | Path, topk: int = 50,
                         batch_size: int = 64, num_workers: int = 2,
                         pair_chunk: int = 50, rerank_models=None,
                         stage1_payload=None, has_ground_truth: bool = True,
                         query_labels: list[str] | None = None,
                         cache_fingerprint: dict | None = None,
                         overwrite: bool = False) -> tuple[dict, bool]:
    """Run X-VLM once and persist the canonical candidate ITM logits.

    The candidate payload order is canonical. Retrieval variants may assign new scores to
    these candidates but must not change their IDs. Returns ``(cache, reused)``.
    ``model`` and ``dataset`` may be None when a valid cache already exists.
    """
    cache_path = Path(cache_path)
    if not overwrite:
        cached = load_valid_rerank_cache(
            cache_path, candidate_payload, topk, cache_fingerprint
        )
        if cached is not None:
            print(f"reusing rerank cache: {cache_path}", flush=True)
            return cached, True
    if model is None or dataset is None:
        raise ValueError("model and dataset are required to build an invalid or missing cache")

    preparation_started = time.perf_counter()
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)
    print(f"[cache 1/3] Encoding eval set | rows={len(dataset):,} "
          f"batch={batch_size}", flush=True)
    enc = encode_eval_set(model, dataset, device, batch_size, num_workers)
    stage1_gallery, stage1_text = load_stage1_features(stage1_payload, enc)
    candidates = load_candidate_payload(candidate_payload, enc)
    if candidates is None:
        raise ValueError("candidate_payload is required for cached reranking")
    topk_idx = candidates["indices"]
    if topk_idx.size(1) != int(topk):
        raise ValueError(f"Aligned candidate width {topk_idx.size(1)} != topk={topk}")

    print(f"[cache 2/3] Cross-encoder ITM | queries={topk_idx.size(0):,} "
          f"K={topk_idx.size(1)} pairs={topk_idx.numel():,}", flush=True)
    itm = itm_rerank(
        model, enc["gallery_embeds"], enc["txt_embeds"], enc["txt_masks"],
        topk_idx, device, pair_chunk, rerank_models=rerank_models,
    )
    payload = _load_torch_payload(candidate_payload)
    query_order = _payload_query_order(payload["query_image_ids"], enc["query_image_ids"])
    candidate_rows = [
        [str(value) for value in payload["candidate_image_ids"][index]]
        for index in query_order
    ]
    metadata = _cache_expected_metadata(candidate_payload, topk, cache_fingerprint)
    metadata.update({
        "gallery_size": len(enc["gallery_ids"]),
        "num_queries": topk_idx.size(0),
        "has_ground_truth": bool(has_ground_truth),
        "created_at_unix": int(time.time()),
        "preparation_seconds": round(time.perf_counter() - preparation_started, 4),
        "peak_vram_mib": {
            f"cuda:{device_index}": round(
                torch.cuda.max_memory_allocated(device_index) / (1024 ** 2), 2
            )
            for device_index in range(torch.cuda.device_count())
        } if torch.cuda.is_available() else {},
    })
    cache = {
        "metadata": metadata,
        "gallery_ids": [str(value) for value in enc["gallery_ids"]],
        "query_image_ids": [str(value) for value in enc["query_image_ids"]],
        "query_labels": [str(value) for value in (
            query_labels if query_labels is not None else enc["query_image_ids"]
        )],
        "candidate_image_ids": candidate_rows,
        "candidate_indices": topk_idx.cpu().long(),
        "pe_scores": candidates["scores"].cpu().float(),
        "itm_logits": itm.cpu().float(),
        "query_feats": stage1_text.cpu().half(),
    }
    if len(cache["query_labels"]) != topk_idx.size(0):
        raise ValueError("query_labels length does not match the query count")
    if has_ground_truth:
        fallback = torch.full(
            (topk_idx.size(0),), len(enc["gallery_ids"]), dtype=torch.long
        )
        cache.update({
            "gt_pos": enc["gt_pos"].cpu().long(),
            "pe_raw_ranks": candidates.get("pe_raw_ranks", fallback).cpu().long(),
            "pe_selected_ranks": candidates.get(
                "pe_selected_ranks", candidates.get("stage1_ranks", fallback)
            ).cpu().long(),
            "fallback_ranks": candidates.get("stage1_ranks", fallback).cpu().long(),
        })
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"[cache 3/3] Saved: {cache_path}", flush=True)
    return cache, False


def _row_minmax(values: Tensor) -> Tensor:
    values = values.float()
    low = values.min(dim=1, keepdim=True).values
    high = values.max(dim=1, keepdim=True).values
    return (values - low) / (high - low).clamp_min(1e-12)


def _row_zscore(values: Tensor) -> Tensor:
    values = values.float()
    return (values - values.mean(dim=1, keepdim=True)) / values.std(
        dim=1, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


def _rank_matrix(values: Tensor) -> Tensor:
    order = values.float().argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(1, values.size(1) + 1).expand_as(order)
    ranks.scatter_(1, order, rank_values)
    return ranks


def _reciprocal_swap_count(order: Tensor, gallery_ids: list[str],
                           query_image_ids: list[str]) -> int:
    query_pos = {str(value): i for i, value in enumerate(query_image_ids)}
    top_ids = [str(gallery_ids[int(row[0])]) for row in order]
    pairs = set()
    for i, value in enumerate(top_ids):
        j = query_pos.get(value)
        if j is not None and j != i and top_ids[j] == str(query_image_ids[i]):
            pairs.add(tuple(sorted((i, j))))
    return len(pairs)


def _transition_counts(reference_order: Tensor, order: Tensor, gt_pos: Tensor) -> dict:
    ref_ok = reference_order[:, 0].eq(gt_pos)
    out_ok = order[:, 0].eq(gt_pos)
    return {
        "both_correct": int((ref_ok & out_ok).sum()),
        "helped": int((~ref_ok & out_ok).sum()),
        "harmed": int((ref_ok & ~out_ok).sum()),
        "both_wrong": int((~ref_ok & ~out_ok).sum()),
    }


def evaluate_cached_rerank(cache: dict, retrieval_scores: Tensor,
                           fusion_family: str = "legacy",
                           fusion_weight: float = 1.0,
                           rank_constant: int = 20,
                           postprocesses=("rerank",),
                           reference_order: Tensor | None = None,
                           include_top10: bool = True) -> dict:
    """Evaluate score fusion and postprocessing using no model forward calls."""
    allowed_fusion = {"legacy", "calibrated", "rank"}
    allowed_post = {"rerank", "greedy_sca", "gale_shapley"}
    if fusion_family not in allowed_fusion:
        raise ValueError(f"fusion_family must be one of {sorted(allowed_fusion)}")
    postprocesses = tuple(dict.fromkeys(postprocesses))
    unknown = set(postprocesses).difference(allowed_post)
    if unknown:
        raise ValueError(f"Unknown postprocesses: {sorted(unknown)}")

    itm = torch.as_tensor(cache["itm_logits"]).float()
    retrieval = torch.as_tensor(retrieval_scores).float()
    indices = torch.as_tensor(cache["candidate_indices"]).long()
    if itm.shape != retrieval.shape or itm.shape != indices.shape:
        raise ValueError("ITM, retrieval, and candidate tensors must have the same shape")

    weight = float(fusion_weight)
    if fusion_family == "legacy":
        itm_component = itm
        retrieval_component = _row_minmax(retrieval)
        final = itm_component + weight * retrieval_component
    elif fusion_family == "calibrated":
        itm_component = _row_zscore(itm)
        retrieval_component = _row_zscore(retrieval)
        final = itm_component + weight * retrieval_component
    else:
        itm_ranks = _rank_matrix(itm).float()
        retrieval_ranks = _rank_matrix(retrieval).float()
        itm_component = 1.0 / (float(rank_constant) + itm_ranks)
        retrieval_component = 1.0 / (float(rank_constant) + retrieval_ranks)
        final = itm_component + weight * retrieval_component

    order_in_k = final.argsort(dim=1, descending=True)
    order_rerank = torch.gather(indices, 1, order_in_k)
    scores_rerank = torch.gather(final, 1, order_in_k)
    gallery_ids = [str(value) for value in cache["gallery_ids"]]
    query_ids = [str(value) for value in cache["query_image_ids"]]
    output = {
        "fusion_family": fusion_family,
        "fusion_weight": weight,
        "rank_constant": int(rank_constant),
        "orders": {"rerank": order_rerank},
        "scores": {"rerank": scores_rerank},
        "top10_by_stage": {},
        "diagnostics": {
            "score_calibration": score_calibration_report(
                itm_component, retrieval_component, final, weight
            ),
            "rerank_top1_conflicts": top1_conflict_count(order_rerank),
            "rerank_reciprocal_swap_pairs": _reciprocal_swap_count(
                order_rerank, gallery_ids, query_ids
            ),
        },
    }
    if include_top10:
        output["top10_by_stage"]["rerank"] = build_top10(
            order_rerank, None, gallery_ids
        )
    has_gt = bool(cache.get("metadata", {}).get("has_ground_truth"))
    if has_gt:
        gt_pos = torch.as_tensor(cache["gt_pos"]).long()
        fallback = torch.as_tensor(cache["fallback_ranks"]).long()
        retrieval_order = torch.gather(
            indices, 1, retrieval.argsort(dim=1, descending=True)
        )
        retrieval_ranks = ranks_after_order(retrieval_order, gt_pos, fallback)
        rerank_ranks = ranks_after_order(order_rerank, gt_pos, fallback)
        output["retrieval"] = report_from_ranks(retrieval_ranks)
        output["rerank"] = report_from_ranks(rerank_ranks)
        output["ranks"] = {"retrieval": retrieval_ranks, "rerank": rerank_ranks}
        output["diagnostics"].update({
            "retrieval_gt_at_rank2": int(retrieval_ranks.eq(2).sum()),
            "rerank_gt_at_rank2": int(rerank_ranks.eq(2).sum()),
        })
        if reference_order is not None:
            output["diagnostics"]["transitions_vs_reference"] = _transition_counts(
                reference_order, order_rerank, gt_pos
            )

    if "greedy_sca" in postprocesses:
        sca_order, _ = greedy_sca(
            order_rerank, scores_rerank,
            query_feats=torch.as_tensor(cache["query_feats"]).float(),
        )
        output["orders"]["greedy_sca"] = sca_order
        output["scores"]["greedy_sca"] = scores_for_order(
            indices, final, sca_order
        )
        if include_top10:
            output["top10_by_stage"]["greedy_sca"] = build_top10(
                sca_order, None, gallery_ids
            )
        output["diagnostics"]["greedy_sca_top1_conflicts"] = top1_conflict_count(
            sca_order
        )
        output["diagnostics"]["greedy_sca_reciprocal_swap_pairs"] = (
            _reciprocal_swap_count(sca_order, gallery_ids, query_ids)
        )
        if has_gt:
            ranks = ranks_after_order(
                sca_order, torch.as_tensor(cache["gt_pos"]).long(),
                torch.as_tensor(cache["fallback_ranks"]).long(),
            )
            output["greedy_sca"] = report_from_ranks(ranks)
            output["ranks"]["greedy_sca"] = ranks
            output["diagnostics"]["greedy_sca_gt_at_rank2"] = int(ranks.eq(2).sum())
            if reference_order is not None:
                output["diagnostics"]["greedy_sca_transitions_vs_reference"] = (
                    _transition_counts(reference_order, sca_order,
                                       torch.as_tensor(cache["gt_pos"]).long())
                )

    if "gale_shapley" in postprocesses:
        matched = gale_shapley_match(order_rerank, scores_rerank)
        if has_gt:
            rerank_ranks = output["ranks"]["rerank"]
            gs_ranks, gs_order = apply_gale_shapley(
                order_rerank, matched, rerank_ranks,
                torch.as_tensor(cache["gt_pos"]).long(),
            )
            output["gale_shapley"] = report_from_ranks(gs_ranks)
            output["ranks"]["gale_shapley"] = gs_ranks
            output["diagnostics"]["gale_shapley_gt_at_rank2"] = int(
                gs_ranks.eq(2).sum()
            )
        else:
            dummy_ranks = torch.ones(order_rerank.size(0), dtype=torch.long)
            dummy_gt = torch.full((order_rerank.size(0),), -1, dtype=torch.long)
            _, gs_order = apply_gale_shapley(
                order_rerank, matched, dummy_ranks, dummy_gt
            )
        output["orders"]["gale_shapley"] = gs_order
        output["scores"]["gale_shapley"] = scores_for_order(
            indices, final, gs_order
        )
        if include_top10:
            output["top10_by_stage"]["gale_shapley"] = build_top10(
                gs_order, None, gallery_ids
            )
        output["diagnostics"]["gale_shapley_top1_conflicts"] = top1_conflict_count(
            gs_order
        )
        output["diagnostics"]["gale_shapley_reciprocal_swap_pairs"] = (
            _reciprocal_swap_count(gs_order, gallery_ids, query_ids)
        )
        if has_gt and reference_order is not None:
            output["diagnostics"]["gale_shapley_transitions_vs_reference"] = (
                _transition_counts(reference_order, gs_order,
                                   torch.as_tensor(cache["gt_pos"]).long())
            )
    return output


# --------------------------------------------------------------------------- orchestrator
@torch.no_grad()
def run_pipeline(model, dataset, device, topk: int = 200, batch_size: int = 64,
                 num_workers: int = 2, use_gale_shapley: bool = True,
                 pair_chunk: int = 50, pairwise_head=None, pairwise_topn: int = 10,
                 use_sca: bool = True, use_sinkhorn: bool = False,
                 sinkhorn_mode: str = "sinkhorn", query_bank_path: str | None = None,
                 sinkhorn_epsilon: float = 0.05, sinkhorn_max_iter: int = 20,
                 rerank_models=None, stage1_payload=None,
                 candidate_payload=None,
                 stage1_weight: float = 1.0, itm_weight: float = 1.0) -> dict:
    """Full inference. Returns stage-wise reports + final top-10 per query.

    `candidate_payload` may provide an ID-aligned, externally ranked Top-K block. Its scores
    are blended with ITM using `stage1_weight`; DBSN must already have been applied upstream.
    If `pairwise_head` (a trained PairwiseHead) is given, a 'pairwise' stage reorders the top-N of
    the ITM order by a round-robin tournament and RRF-fuses it back — measured independently.
    """
    print(f"[1/5] Encoding eval set | rows={len(dataset):,} batch={batch_size}", flush=True)
    enc = encode_eval_set(model, dataset, device, batch_size, num_workers)
    print(f"      encoded gallery={len(enc['gallery_ids']):,} queries={enc['txt_feats'].size(0):,}", flush=True)
    print("[2/5] Computing or loading Stage-1 candidates", flush=True)
    stage1_gallery, stage1_text = load_stage1_features(stage1_payload, enc)
    candidates = load_candidate_payload(candidate_payload, enc)
    if candidates is None:
        sim_raw = stage1_text @ stage1_gallery.t()
        ranks_raw = stage1_ranks(sim_raw, enc["gt_pos"])
        sim = sim_raw
        if use_sinkhorn:
            print(f"      applying {sinkhorn_mode} normalization "
                  f"(epsilon={sinkhorn_epsilon}, iter={sinkhorn_max_iter})", flush=True)
            sim = apply_sinkhorn_or_dbsn(sim, stage1_gallery,
                                         query_bank_path=query_bank_path,
                                         mode=sinkhorn_mode,
                                         epsilon=sinkhorn_epsilon,
                                         max_iter=sinkhorn_max_iter)
        ranks1 = stage1_ranks(sim, enc["gt_pos"])
        K = min(topk, sim.size(1))
        topk_sim, topk_idx = sim.topk(K, dim=1)
    else:
        if use_sinkhorn:
            raise ValueError(
                "candidate_payload is already ranked; apply PE DBSN during candidate "
                "generation and pass use_sinkhorn=False to the X-VLM pipeline"
            )
        sim = None
        topk_idx = candidates["indices"]
        topk_sim = candidates["scores"]
        K = topk_idx.size(1)
        if topk != K:
            print(f"      candidate payload fixes K={K}; ignoring requested topk={topk}",
                  flush=True)
        fallback = torch.full((topk_idx.size(0),), len(enc["gallery_ids"]), dtype=torch.long)
        ranks_raw = candidates.get("pe_raw_ranks", fallback)
        ranks_pe_selected = candidates.get("pe_selected_ranks", ranks_raw)
        ranks1 = candidates.get("stage1_ranks", ranks_after_order(
            topk_idx, enc["gt_pos"], fallback
        ))
        metadata = candidates.get("metadata", {})
        print(f"      loaded candidate payload | mode={metadata.get('mode', 'unknown')} "
              f"Q={topk_idx.size(0):,} K={K}", flush=True)

    if candidates is None:
        ranks_pe_selected = ranks1
    rep_raw = report_from_ranks(ranks_raw)
    rep_pe_selected = report_from_ranks(ranks_pe_selected)
    rep1 = report_from_ranks(ranks1)
    print(f"[3/5] Cross-encoder ITM rerank | queries={topk_idx.size(0):,} K={K} "
          f"pairs={topk_idx.size(0) * K:,} chunk={pair_chunk}", flush=True)
    itm = itm_rerank(model, enc["gallery_embeds"], enc["txt_embeds"], enc["txt_masks"],
                     topk_idx, device, pair_chunk, rerank_models=rerank_models)
    final = itm_weight * itm + stage1_weight * topk_sim
    score_calibration = score_calibration_report(itm, topk_sim, final, stage1_weight)
    ranks2, order2 = ranks_after_rerank(sim, topk_idx, final, enc["gt_pos"], ranks1)
    rep2 = report_from_ranks(ranks2)
    reports = {
        "stage1_raw": rep_raw,
        "pe_selected": rep_pe_selected,
        "stage1": rep1,
        "rerank": rep2,
    }
    ranks_out = {
        "stage1_raw": ranks_raw,
        "pe_selected": ranks_pe_selected,
        "stage1": ranks1,
        "rerank": ranks2,
    }

    # (pairwise / duo) reorder the top-N of the ITM order via the comparator, RRF-fused
    order_pw, ranks_pw = order2, ranks2
    if pairwise_head is not None:
        print("[4/5] Pairwise rerank", flush=True)
        N = min(pairwise_topn, K)
        feats = pairwise_features(model, enc["gallery_embeds"], enc["txt_embeds"],
                                  enc["txt_masks"], order2[:, :N], device)
        order_pw = order2.clone()
        ranks_pw = ranks2.clone()
        for q in range(order2.size(0)):
            itm_order = order2[q].tolist()
            perm = pairwise_rerank(pairwise_head, feats[q]).tolist()
            pw_topn = [itm_order[p] for p in perm]
            fused_topn = rrf_fuse([pw_topn, itm_order[:N]])              # blend duo + ITM on top-N
            full = fused_topn + itm_order[N:]
            order_pw[q] = torch.tensor(full, dtype=order2.dtype)
            hit = (order_pw[q] == enc["gt_pos"][q]).nonzero(as_tuple=True)[0]
            ranks_pw[q] = int(hit[0]) + 1 if len(hit) else int(ranks2[q])
        reports["pairwise"] = report_from_ranks(ranks_pw)
        ranks_out["pairwise"] = ranks_pw

    base_order, base_ranks = order_pw, ranks_pw          # GS runs on the best order so far
    base_scores = scores_for_order(topk_idx, final, base_order)

    if use_sca:
        print("[4/5] Greedy SCA ablation", flush=True)
        order_sca, _ = greedy_sca(base_order, base_scores, query_feats=stage1_text)
        ranks_sca = ranks_after_order(order_sca, enc["gt_pos"], ranks1)
        reports["greedy_sca"] = report_from_ranks(ranks_sca)
        ranks_out["greedy_sca"] = ranks_sca
    else:
        order_sca = base_order

    if use_gale_shapley:
        print("[5/5] Gale-Shapley stable matching", flush=True)
        matched = gale_shapley_match(base_order, base_scores)
        ranks3, order3 = apply_gale_shapley(base_order, matched, base_ranks, enc["gt_pos"])
        rep3 = report_from_ranks(ranks3)
    else:
        ranks3, order3, rep3 = base_ranks, base_order, report_from_ranks(base_ranks)
    reports["gale_shapley"] = rep3
    ranks_out["final"] = ranks3

    # top-10 = reranked K-block first, then the stage-1 tail (items outside the block keep
    print("      building final top-10", flush=True)
    # their cosine order) — matters when topk < 10 and is the correct general semantics
    top10_by_stage = {
        "rerank": build_top10(order2, sim, enc["gallery_ids"]),
        "greedy_sca": build_top10(order_sca, sim, enc["gallery_ids"]),
        "gale_shapley": build_top10(order3, sim, enc["gallery_ids"]),
    }
    top10 = top10_by_stage["gale_shapley" if use_gale_shapley else "rerank"]
    diagnostics = {
        "stage1_gt_at_rank2": int((ranks1 == 2).sum()),
        "rerank_gt_at_rank2": int((ranks2 == 2).sum()),
        "greedy_sca_gt_at_rank2": int((ranks_sca == 2).sum()) if use_sca else int((ranks2 == 2).sum()),
        "gale_shapley_gt_at_rank2": int((ranks3 == 2).sum()),
        "rerank_top1_conflicts": top1_conflict_count(order2),
        "greedy_sca_top1_conflicts": top1_conflict_count(order_sca),
        "gale_shapley_top1_conflicts": top1_conflict_count(order3),
        "score_calibration": score_calibration,
    }
    return dict(**reports, ranks=ranks_out,
                top10=top10, top10_by_stage=top10_by_stage,
                diagnostics=diagnostics,
                gallery_size=len(enc["gallery_ids"]), num_queries=topk_idx.size(0), topk=K)


@torch.no_grad()
def run_submit_pipeline(model, dataset, device, topk: int = 200, batch_size: int = 64,
                        num_workers: int = 2, pair_chunk: int = 50,
                        postprocess: str = "gale_shapley",
                        use_sinkhorn: bool = False,
                        sinkhorn_mode: str = "sinkhorn",
                        query_bank_path: str | None = None,
                        sinkhorn_epsilon: float = 0.05,
                        sinkhorn_max_iter: int = 20,
                        rerank_models=None, stage1_payload=None,
                        candidate_payload=None,
                        stage1_weight: float = 1.0, itm_weight: float = 1.0) -> dict:
    """No-GT inference path for the official hidden/test set.

    Returns top-10 image ids without computing rank metrics. `postprocess` is one of:
    "rerank", "greedy_sca", "gale_shapley".
    """
    if postprocess not in {"rerank", "greedy_sca", "gale_shapley"}:
        raise ValueError("postprocess must be one of: rerank, greedy_sca, gale_shapley")

    print(f"[1/5] Encoding submit set | rows={len(dataset):,} batch={batch_size}", flush=True)
    enc = encode_eval_set(model, dataset, device, batch_size, num_workers)
    print(f"      encoded gallery={len(enc['gallery_ids']):,} queries={enc['txt_feats'].size(0):,}", flush=True)
    print("[2/5] Computing or loading Stage-1 candidates", flush=True)
    stage1_gallery, stage1_text = load_stage1_features(stage1_payload, enc)
    candidates = load_candidate_payload(candidate_payload, enc)
    if candidates is None:
        sim = stage1_text @ stage1_gallery.t()
        if use_sinkhorn:
            print(f"      applying {sinkhorn_mode} normalization "
                  f"(epsilon={sinkhorn_epsilon}, iter={sinkhorn_max_iter})", flush=True)
            sim = apply_sinkhorn_or_dbsn(sim, stage1_gallery,
                                         query_bank_path=query_bank_path,
                                         mode=sinkhorn_mode,
                                         epsilon=sinkhorn_epsilon,
                                         max_iter=sinkhorn_max_iter)
        K = min(topk, sim.size(1))
        topk_sim, topk_idx = sim.topk(K, dim=1)
    else:
        if use_sinkhorn:
            raise ValueError(
                "candidate_payload is already ranked; apply PE DBSN during candidate "
                "generation and pass use_sinkhorn=False to the X-VLM pipeline"
            )
        sim = None
        topk_idx = candidates["indices"]
        topk_sim = candidates["scores"]
        K = topk_idx.size(1)
        metadata = candidates.get("metadata", {})
        print(f"      loaded candidate payload | mode={metadata.get('mode', 'unknown')} "
              f"Q={topk_idx.size(0):,} K={K}", flush=True)

    print(f"[3/5] Cross-encoder ITM rerank | queries={topk_idx.size(0):,} K={K} "
          f"pairs={topk_idx.size(0) * K:,} chunk={pair_chunk}", flush=True)
    itm = itm_rerank(model, enc["gallery_embeds"], enc["txt_embeds"], enc["txt_masks"],
                     topk_idx, device, pair_chunk, rerank_models=rerank_models)
    final = itm_weight * itm + stage1_weight * topk_sim
    score_calibration = score_calibration_report(itm, topk_sim, final, stage1_weight)
    order_in_k = final.argsort(dim=1, descending=True)
    order_rerank = torch.gather(topk_idx, 1, order_in_k)
    scores_rerank = torch.gather(final, 1, order_in_k)

    print("[4/5] Greedy SCA candidate", flush=True)
    order_sca, _ = greedy_sca(order_rerank, scores_rerank, query_feats=stage1_text)

    print("[5/5] Gale-Shapley candidate", flush=True)
    matched = gale_shapley_match(order_rerank, scores_rerank)
    dummy_ranks = torch.ones(order_rerank.size(0), dtype=torch.long)
    dummy_gt = torch.full((order_rerank.size(0),), -1, dtype=torch.long)
    _, order_gs = apply_gale_shapley(order_rerank, matched, dummy_ranks, dummy_gt)

    top10_by_stage = {
        "rerank": build_top10(order_rerank, sim, enc["gallery_ids"]),
        "greedy_sca": build_top10(order_sca, sim, enc["gallery_ids"]),
        "gale_shapley": build_top10(order_gs, sim, enc["gallery_ids"]),
    }
    return {
        "top10": top10_by_stage[postprocess],
        "top10_by_stage": top10_by_stage,
        "postprocess": postprocess,
        "gallery_size": len(enc["gallery_ids"]),
        "num_queries": topk_idx.size(0),
        "topk": K,
        "diagnostics": {"score_calibration": score_calibration},
    }
