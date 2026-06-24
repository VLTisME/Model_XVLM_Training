"""VAL-B retrieval evaluation: encode queries + gallery, score, report mAP/MRR/R@K.

Review fix #3: the old version assumed query i <-> image i (a clean N x N diagonal task), which
CANNOT represent the real condition — 1,978 queries vs a 36,773-image gallery with 34,795
distractors. Checkpoint selection on an easy 1:1 task does not reflect the test.

Now we decouple queries from the gallery via `image_id`:
  - the gallery is every UNIQUE image_id in the eval set (so the data team can add distractor rows,
    i.e. rows that carry an image but no caption -> gallery-only, never a query);
  - each query's ground truth is the gallery position of ITS OWN image_id.

`assemble_query_gallery` is the pure, unit-tested core; `evaluate_retrieval` just feeds it encoded
features. This is the bi-encoder (ITC) eval used for checkpoint selection; the full inference
post-processing (Sinkhorn / cross-encoder rerank / ensemble / SCA) lives in a separate repo.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..metrics import full_report


def assemble_query_gallery(
    img_feats: Tensor,
    txt_feats: Tensor,
    image_ids: list,
    is_query: list[bool],
) -> tuple[Tensor, Tensor]:
    """Build the [Q, G] similarity matrix and the ground-truth column per query.

    Args:
        img_feats: [R, d] per-row image features (one per eval row).
        txt_feats: [R, d] per-row text features.
        image_ids: length-R image identity per row (distractor rows have their own unique id).
        is_query:  length-R bool; a row is a query iff it has a caption.
    Returns:
        sim [Q, G], gt_index [Q]  (column index of each query's true image in the gallery).
    """
    id_to_pos: dict = {}
    gal_rows: list[int] = []
    for r, iid in enumerate(image_ids):
        if iid not in id_to_pos:
            id_to_pos[iid] = len(gal_rows)
            gal_rows.append(r)
    gallery = img_feats[gal_rows]                                  # [G, d] unique images
    q_rows = [r for r, q in enumerate(is_query) if q]
    txt = txt_feats[q_rows]                                        # [Q, d]
    gt_index = torch.tensor([id_to_pos[image_ids[r]] for r in q_rows], device=img_feats.device)
    sim = txt @ gallery.t()                                        # [Q, G]
    return sim, gt_index


@torch.no_grad()
def evaluate_retrieval(model, dataset, device, batch_size: int = 64, num_workers: int = 4) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=_eval_collate)
    img_feats, txt_feats, image_ids, is_query = [], [], [], []
    for batch in loader:
        kpts = batch.get("keypoints")
        fv, ft = model.encode_for_eval(batch["image"].to(device),
                                       batch["input_ids"].to(device),
                                       batch["attention_mask"].to(device),
                                       keypoints=kpts.to(device) if kpts is not None else None)
        img_feats.append(fv.float().cpu())
        txt_feats.append(ft.float().cpu())
        image_ids.extend(batch["image_id"])
        is_query.extend(batch["is_query"])
    sim, gt_index = assemble_query_gallery(torch.cat(img_feats), torch.cat(txt_feats),
                                           image_ids, is_query)
    return full_report(sim, gt_index, ks=(1, 5, 10, 50, 200))


def _eval_collate(batch):
    out = {
        "image": torch.stack([b["image"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "image_id": [b["image_id"] for b in batch],
        "is_query": [b["is_query"] for b in batch],
    }
    # pose branch needs keypoints at eval too (train/eval consistency)
    if all("keypoints" in b for b in batch):
        out["keypoints"] = torch.stack([b["keypoints"] for b in batch])
    return out
