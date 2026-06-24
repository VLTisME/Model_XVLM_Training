"""BBox-aware, pair-consistent sim-to-real augmentation and deterministic eval resize."""
from __future__ import annotations

import io
import random
from dataclasses import dataclass

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class AugmentSpec:
    blur_mode: str | None = None
    blur_kernel: int = 3
    blur_horizontal: bool = True
    jpeg_quality: int | None = None
    downscale: float | None = None
    jitter: tuple[float, float, float, float] | None = None
    noise_std: float | None = None
    erase: tuple[float, float, float, float] | None = None


def _expand_bbox_xywh(bbox, margin: float = 0.125):
    if bbox is None:
        return None
    x, y, w, h = [float(v) for v in bbox]
    return (
        max(0.0, x - margin * w),
        max(0.0, y - margin * h),
        min(1.0, x + w + margin * w),
        min(1.0, y + h + margin * h),
    )


def _motion_blur(image: Image.Image, kernel: int, horizontal: bool) -> Image.Image:
    # Pillow's generic Kernel filter is reliable here only for 3x3 and 5x5 kernels.
    kernel = 3 if int(kernel) <= 3 else 5
    weights = [0.0] * (kernel * kernel)
    if horizontal:
        row = kernel // 2
        for col in range(kernel):
            weights[row * kernel + col] = 1.0 / kernel
    else:
        col = kernel // 2
        for row in range(kernel):
            weights[row * kernel + col] = 1.0 / kernel
    return image.filter(ImageFilter.Kernel((kernel, kernel), weights, scale=1.0))


def _masked_blur(image: Image.Image, bbox, mode: str, kernel: int, horizontal: bool) -> Image.Image:
    blurred = _motion_blur(image, kernel, horizontal)
    expanded = _expand_bbox_xywh(bbox)
    if mode == "global" or expanded is None:
        return blurred
    x1, y1, x2, y2 = expanded
    width, height = image.size
    box = (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )
    mask = Image.new("L", image.size, 0 if mode == "subject" else 255)
    mask.paste(255 if mode == "subject" else 0, box)
    return Image.composite(blurred, image, mask)


class BBoxAwareTransform:
    """Global resize plus at most two corruption families.

    `sample_spec()` is separate from `apply()` so both endpoints of a hard pair can share
    exactly the same corruption family and severity.
    """

    def __init__(
        self,
        size: int = 384,
        enabled: bool = True,
        motion_blur_p: float = 0.25,
        jpeg_p: float = 0.35,
        downscale_p: float = 0.30,
        color_jitter_p: float = 0.30,
        noise_p: float = 0.15,
        erase_p: float = 0.20,
        max_ops: int = 2,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ):
        self.size = int(size)
        self.enabled = enabled
        self.probabilities = {
            "blur": motion_blur_p,
            "jpeg": jpeg_p,
            "downscale": downscale_p,
            "jitter": color_jitter_p,
            "noise": noise_p,
            "erase": erase_p,
        }
        self.max_ops = int(max_ops)
        self._normalize = T.Normalize(mean, std)

    def sample_spec(self, rng: random.Random | None = None) -> AugmentSpec:
        rng = rng or random
        if not self.enabled:
            return AugmentSpec()
        selected = [name for name, probability in self.probabilities.items() if rng.random() < probability]
        rng.shuffle(selected)
        selected = selected[: self.max_ops]
        values = {}
        if "blur" in selected:
            roll = rng.random()
            values["blur_mode"] = "subject" if roll < 0.45 else "background" if roll < 0.80 else "global"
            values["blur_kernel"] = rng.choice((3, 5))
            values["blur_horizontal"] = rng.random() < 0.5
        if "jpeg" in selected:
            values["jpeg_quality"] = rng.randint(40, 95)
        if "downscale" in selected:
            values["downscale"] = rng.uniform(0.5, 0.9)
        if "jitter" in selected:
            values["jitter"] = (
                rng.uniform(0.8, 1.2),
                rng.uniform(0.8, 1.2),
                rng.uniform(0.8, 1.2),
                rng.uniform(-0.05, 0.05),
            )
        if "noise" in selected:
            values["noise_std"] = rng.uniform(0.005, 0.03)
        if "erase" in selected:
            area = rng.uniform(0.02, 0.15)
            aspect = rng.uniform(0.5, 2.0)
            h = min(0.6, (area / aspect) ** 0.5)
            w = min(0.6, (area * aspect) ** 0.5)
            values["erase"] = (rng.uniform(0, 1 - w), rng.uniform(0, 1 - h), w, h)
        return AugmentSpec(**values)

    def apply(self, image: Image.Image, bbox=None, spec: AugmentSpec | None = None) -> torch.Tensor:
        image = image.convert("RGB")
        spec = spec or self.sample_spec()
        if spec.blur_mode:
            image = _masked_blur(
                image, bbox, spec.blur_mode, spec.blur_kernel, spec.blur_horizontal
            )
        if spec.jpeg_quality is not None:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=spec.jpeg_quality)
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                image = decoded.convert("RGB")
        if spec.downscale is not None:
            width, height = image.size
            small = (
                max(1, int(round(width * spec.downscale))),
                max(1, int(round(height * spec.downscale))),
            )
            image = image.resize(small, Image.Resampling.BILINEAR).resize(
                (width, height), Image.Resampling.BILINEAR
            )
        if spec.jitter is not None:
            brightness, contrast, saturation, hue = spec.jitter
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            image = TF.adjust_hue(image, hue)

        tensor = TF.to_tensor(TF.resize(image, [self.size, self.size], antialias=True))
        if spec.noise_std is not None:
            tensor = (tensor + torch.randn_like(tensor) * spec.noise_std).clamp(0, 1)
        if spec.erase is not None:
            x, y, w, h = spec.erase
            tensor = TF.erase(
                tensor,
                int(round(y * self.size)),
                int(round(x * self.size)),
                max(1, int(round(h * self.size))),
                max(1, int(round(w * self.size))),
                0.0,
            )
        return self._normalize(tensor)

    def __call__(self, image: Image.Image, bbox=None):
        return self.apply(image, bbox, self.sample_spec())


class LHPTransform(BBoxAwareTransform):
    """Compatibility alias for existing configs/imports."""

    def __init__(
        self,
        size: int = 384,
        min_scale: float = 0.5,
        use_bbox: bool = True,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(size=size, enabled=enabled, **kwargs)
        self.min_scale = min_scale
        self.use_bbox = use_bbox


def build_eval_transform(size: int = 384):
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
