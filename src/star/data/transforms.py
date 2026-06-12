"""LHP — Local-global Hybrid augmentation (train only) + plain eval transform.

analyze.md §10. Per image: p ~ N(0.5, 1/6).
  p > 0.5 -> LOCAL  : person-bbox RandomResizedCrop (scale >= 0.5), resize to S
  else    -> GLOBAL : full image resize to S
Inference uses GLOBAL only (build_eval_transform) so no detail is lost at scoring time.

Safety (answers the "won't local crop lose detail?" concern):
  - stochastic: GLOBAL is also seen across epochs
  - scale >= 0.5 (never an aggressive crop)
  - crop centered on the person bbox (keeps the subject), fallback to center crop
"""
from __future__ import annotations

import random

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normal_p(mean: float = 0.5, var: float = 1.0 / 6.0) -> float:
    return random.gauss(mean, var ** 0.5)


def _bbox_crop(img: Image.Image, bbox, min_scale: float) -> Image.Image:
    """Crop around a person bbox (normalized [x, y, w, h]) with random scale >= min_scale.

    The crop is expanded around the bbox center; if bbox is missing we fall back to a
    standard RandomResizedCrop region (handled by the caller).
    """
    W, H = img.size
    x, y, w, h = bbox
    cx, cy = (x + w / 2) * W, (y + h / 2) * H
    scale = random.uniform(min_scale, 1.0)
    # crop side proportional to image, but at least covering the bbox
    side = max(w * W, h * H, scale * min(W, H))
    half = side / 2
    left = int(max(0, min(cx - half, W - side)))
    top = int(max(0, min(cy - half, H - side)))
    side = int(min(side, W - left, H - top))
    return img.crop((left, top, left + side, top + side))


class LHPTransform:
    """Callable transform. Pass the optional normalized bbox to enable person-aware local crop."""

    def __init__(self, size: int = 384, min_scale: float = 0.5, use_bbox: bool = True, enabled: bool = True):
        self.size = size
        self.min_scale = min_scale
        self.use_bbox = use_bbox
        self.enabled = enabled
        self._rrc = T.RandomResizedCrop(size, scale=(min_scale, 1.0), ratio=(0.75, 1.3333))
        self._to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self._global = T.Resize((size, size))

    def __call__(self, img: Image.Image, bbox=None):
        img = img.convert("RGB")
        if self.enabled and _normal_p() > 0.5:
            # LOCAL view
            if self.use_bbox and bbox is not None:
                crop = _bbox_crop(img, bbox, self.min_scale)
                crop = TF.resize(crop, [self.size, self.size])
            else:
                crop = self._rrc(img)
            crop = TF.hflip(crop) if random.random() < 0.5 else crop
            return self._to_tensor(crop)
        # GLOBAL view
        return self._to_tensor(self._global(img))


def build_eval_transform(size: int = 384):
    """Deterministic transform used at validation / inference (GLOBAL full image)."""
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
