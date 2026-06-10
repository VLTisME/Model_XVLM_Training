"""Evaluator core (review fix #3): gallery decoupled from queries, distractor-aware."""
import torch
import torch.nn.functional as F

from star.engine import assemble_query_gallery
from star.metrics import full_report


def test_distractor_rows_join_gallery_not_queries():
    # 3 query rows (own image is GT) + 2 distractor rows (image-only, unique ids)
    d = 16
    base = F.normalize(torch.randn(3, d), dim=-1)
    img = torch.cat([base, F.normalize(torch.randn(2, d), dim=-1)])   # 5 images
    txt = torch.cat([base, torch.zeros(2, d)])                       # queries match their image
    image_ids = ["a", "b", "c", "dist1", "dist2"]
    is_query = [True, True, True, False, False]

    sim, gt = assemble_query_gallery(img, txt, image_ids, is_query)
    assert sim.shape == (3, 5)            # 3 queries x 5-image gallery (incl. 2 distractors)
    assert gt.tolist() == [0, 1, 2]       # each query's GT is its own image position
    # perfect features -> every GT ranked #1 -> mAP == 1
    rep = full_report(sim, gt)
    assert abs(rep["mAP"] - 1.0) < 1e-6 and abs(rep["R@1"] - 1.0) < 1e-6


def test_shared_image_dedup_in_gallery():
    # two captions of the SAME image -> one gallery entry, both queries point to it
    d = 8
    v = F.normalize(torch.randn(1, d), dim=-1)
    img = torch.cat([v, v, F.normalize(torch.randn(1, d), dim=-1)])
    txt = torch.cat([v, v, F.normalize(torch.randn(1, d), dim=-1)])
    image_ids = ["img0", "img0", "img1"]      # rows 0,1 share the image
    is_query = [True, True, True]
    sim, gt = assemble_query_gallery(img, txt, image_ids, is_query)
    assert sim.shape == (3, 2)                # gallery deduped to 2 unique images
    assert gt.tolist() == [0, 0, 1]


def test_distractor_can_steal_rank():
    # a distractor more similar than the GT pushes the GT to rank 2 -> mAP = 0.5 for that query
    d = 4
    q = F.normalize(torch.tensor([[1.0, 0, 0, 0]]), dim=-1)
    gt_img = F.normalize(torch.tensor([[0.6, 0.1, 0, 0]]), dim=-1)
    distractor = F.normalize(torch.tensor([[0.9, 0.1, 0, 0]]), dim=-1)   # closer to q than GT
    img = torch.cat([gt_img, distractor])
    txt = q
    sim, gt = assemble_query_gallery(img, txt, ["gt", "dist"], [True, False])
    rep = full_report(sim, gt)
    assert abs(rep["mAP"] - 0.5) < 1e-6      # GT at rank 2 => AP = 1/2
