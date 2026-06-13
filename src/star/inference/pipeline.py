"""STAR-v3 inference pipeline (per inference2.svg, trimmed to the agreed core):

    encode (GLOBAL images, no LHP, no pose) -> Stage-1 ITC cosine -> (2) Top-K filter
        -> (3) cross-encoder ITM re-rank (BLIP-style: itm_logit + cosine)
        -> (5) Gale-Shapley 1-1 rank-1 assignment -> top-10 / query

Dropped per decision: (1) Sinkhorn/DBSN, (4) ensemble, GNN/k-reciprocal (optional), ViTPose.
Metrics are reported at EVERY stage (cosine / +rerank / +GS) so each block's contribution
is measured, not assumed.

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

import torch
from torch import Tensor
from torch.utils.data import DataLoader


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

    for batch in loader:
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
                txt_feats=txt_feats, txt_embeds=txt_embeds, txt_masks=masks, gt_pos=gt_pos)


# --------------------------------------------------------------------------- stage 1
def stage1_ranks(sim: Tensor, gt_pos: Tensor) -> Tensor:
    """1-based rank of GT under cosine scores (pessimistic on ties, same as star.metrics)."""
    q = torch.arange(sim.size(0))
    gt_score = sim[q, gt_pos].unsqueeze(1)
    greater = (sim > gt_score).sum(dim=1)
    ties = (sim == gt_score).sum(dim=1) - 1
    return greater + ties + 1


# --------------------------------------------------------------------------- (3) rerank
@torch.no_grad()
def itm_rerank(model, gallery_embeds: Tensor, txt_embeds: Tensor, txt_masks: Tensor,
               topk_idx: Tensor, device, pair_chunk: int = 50) -> Tensor:
    """ITM logit[:,1] for each (query, top-K image) pair. Returns [Q, K] fp32 cpu."""
    model.eval()
    use_amp = "cuda" in str(device)
    Q, K = topk_idx.shape
    out = torch.empty(Q, K)
    for qi in range(Q):
        t_emb = txt_embeds[qi].to(device)
        t_mask = txt_masks[qi].to(device)
        idx = topk_idx[qi]
        for s in range(0, K, pair_chunk):
            sel = idx[s:s + pair_chunk]
            img = gallery_embeds[sel].to(device).float()                 # [c, Ni, H]
            te = t_emb.unsqueeze(0).expand(img.size(0), -1, -1)
            tm = t_mask.unsqueeze(0).expand(img.size(0), -1)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model.backbone.itm_logits(img, te, tm)          # [c, 2]
            out[qi, s:s + len(sel)] = logits.float()[:, 1].cpu()
    return out


def ranks_after_rerank(sim: Tensor, topk_idx: Tensor, final_scores: Tensor,
                       gt_pos: Tensor, ranks_s1: Tensor) -> tuple[Tensor, Tensor]:
    """Exact GT ranks after reordering the K-block by final_scores (desc).

    Returns (ranks [Q], order [Q, K] = topk_idx reordered by the new scores).
    """
    order_in_k = final_scores.argsort(dim=1, descending=True)            # [Q, K]
    new_order = torch.gather(topk_idx, 1, order_in_k)                    # gallery idx, reranked
    ranks = ranks_s1.clone()
    for qi in range(sim.size(0)):
        hit = (new_order[qi] == gt_pos[qi]).nonzero(as_tuple=True)[0]
        if len(hit):
            ranks[qi] = int(hit[0]) + 1          # GT inside K-block -> new position
        # else: only the K items above it were permuted -> stage-1 rank unchanged
    return ranks, new_order


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


# --------------------------------------------------------------------------- reporting
def report_from_ranks(ranks: Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    r = ranks.float()
    rep = {"mAP": float((1.0 / r).mean()), "MRR": float((1.0 / r).mean())}
    for k in ks:
        rep[f"R@{k}"] = float((r <= k).float().mean())
    return rep


# --------------------------------------------------------------------------- orchestrator
@torch.no_grad()
def run_pipeline(model, dataset, device, topk: int = 100, batch_size: int = 64,
                 num_workers: int = 2, use_gale_shapley: bool = True,
                 pair_chunk: int = 50) -> dict:
    """Full inference. Returns stage-wise reports + final top-10 per query."""
    enc = encode_eval_set(model, dataset, device, batch_size, num_workers)
    sim = enc["txt_feats"] @ enc["gallery_feats"].t()                    # [Q, G]
    K = min(topk, sim.size(1))

    ranks1 = stage1_ranks(sim, enc["gt_pos"])
    rep1 = report_from_ranks(ranks1)

    topk_sim, topk_idx = sim.topk(K, dim=1)
    itm = itm_rerank(model, enc["gallery_embeds"], enc["txt_embeds"], enc["txt_masks"],
                     topk_idx, device, pair_chunk)
    final = itm + topk_sim                                                # BLIP: logit + cosine
    ranks2, order2 = ranks_after_rerank(sim, topk_idx, final, enc["gt_pos"], ranks1)
    rep2 = report_from_ranks(ranks2)

    if use_gale_shapley:
        scores2 = torch.gather(final, 1, final.argsort(dim=1, descending=True))
        matched = gale_shapley_match(order2, scores2)
        ranks3, order3 = apply_gale_shapley(order2, matched, ranks2, enc["gt_pos"])
        rep3 = report_from_ranks(ranks3)
    else:
        ranks3, order3, rep3 = ranks2, order2, rep2

    # top-10 = reranked K-block first, then the stage-1 tail (items outside the block keep
    # their cosine order) — matters when topk < 10 and is the correct general semantics
    s1_order = sim.argsort(dim=1, descending=True)
    top10 = []
    for q in range(order3.size(0)):
        seen = order3[q].tolist()
        tail = [int(g) for g in s1_order[q] if int(g) not in set(seen)]
        full = (seen + tail)[:10]
        top10.append([enc["gallery_ids"][g] for g in full])
    return dict(stage1=rep1, rerank=rep2, gale_shapley=rep3,
                ranks={"stage1": ranks1, "rerank": ranks2, "final": ranks3},
                top10=top10, gallery_size=sim.size(1), num_queries=sim.size(0), topk=K)
