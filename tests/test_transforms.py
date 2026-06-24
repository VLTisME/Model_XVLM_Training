"""LHP transform + LoRA injection/merge correctness."""
import torch
from PIL import Image

from star.data import PairAwareCollator
from star.data.transforms import AugmentSpec, LHPTransform, build_eval_transform
from star.models.lora import LoRALinear, merge_lora


def _img():
    return Image.new("RGB", (640, 480), (120, 100, 80))


def test_eval_transform_shape():
    t = build_eval_transform(384)
    out = t(_img())
    assert out.shape == (3, 384, 384)


def test_lhp_global_branch_shape():
    t = LHPTransform(size=384, enabled=True)
    out = t(_img(), bbox=None)
    assert out.shape == (3, 384, 384)


def test_lhp_local_bbox_branch_shape():
    t = LHPTransform(size=256, enabled=True, use_bbox=True)
    out = t(_img(), bbox=[0.3, 0.3, 0.4, 0.4])
    assert out.shape == (3, 256, 256)


def test_motion_blur_maps_large_requested_kernel_to_supported_size():
    t = LHPTransform(size=64, enabled=True)
    out = t.apply(
        _img(),
        bbox=None,
        spec=AugmentSpec(
            blur_mode="global",
            blur_kernel=7,
            blur_horizontal=True,
        ),
    )
    assert out.shape == (3, 64, 64)


def test_pair_collator_builds_symmetric_partner_indices():
    transform = LHPTransform(size=64, enabled=True)
    collator = PairAwareCollator(transform, hard_pairs=2)
    batch = [
        {
            "image": _img(),
            "bbox": [0.2, 0.2, 0.5, 0.5],
            "input_ids": torch.ones(8, dtype=torch.long),
            "attention_mask": torch.ones(8, dtype=torch.long),
            "instance_id": i,
            "row_index": i,
        }
        for i in range(4)
    ]
    out = collator(batch)
    assert out["image"].shape == (4, 3, 64, 64)
    assert out["partner_index"].tolist() == [1, 0, 3, 2]


def test_lora_starts_as_identity_then_merges():
    base = torch.nn.Linear(16, 8)
    x = torch.randn(4, 16)
    lora = LoRALinear(base, r=4, alpha=8)
    lora.eval()  # merge is an inference op; compare with dropout disabled
    # B initialized to zero => delta is zero => output equals base at init
    assert torch.allclose(lora(x), base(x), atol=1e-6)
    # after a fake update to B, merge must fold the delta into the base weight
    with torch.no_grad():
        lora.lora_B.add_(0.1)
    before = lora(x)
    merge_lora(lora)
    after = lora(x)
    assert torch.allclose(before, after, atol=1e-5)
    assert lora.merged
