"""Legacy pipeline reproduction: the transform that produced every historical result.

RECOVERED, HIGH confidence, verified pixel-exactly.

`feature_extractor_folder.py` claims to fuse "ViT features" with the RGB image.
It does not. `ViTImageProcessor` returns preprocessed `pixel_values`, and the
subsequent PIL round-trip converts a [-1, 1] float tensor with
`(npimg * 255).astype(np.uint8)` -- no clamp -- so negative values wrap modulo
256, twice. The net effect is a fixed, channel-identical, non-monotonic 256-entry
lookup table applied pointwise, then blended with the original.

Verification: reproducing this function and comparing against the surviving
`dataset/features_extracted_dataset-(0.2-0.8)/` gives mean absolute error
**0.0000** with exact byte equality, at alpha=0.2, beta=0.8. The same comparison
against `features_extracted_dataset-org/` matches at alpha=0.0 (i.e. that
directory is the plain resized original, despite its name).

This module exists so the historical arm (A2) is reproducible. It is not a
recommended transform.
"""

from __future__ import annotations

import functools

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_round_trip = transforms.Compose(
    [
        transforms.ToPILImage(),                                   # unclamped uint8 wrap #1
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        transforms.ToPILImage(),                                   # unclamped uint8 wrap #2
    ]
)


def _vit_pixel_values(pil_img: Image.Image, size: int = 224) -> torch.Tensor:
    """ViTImageProcessor(vit-base-patch16-224-in21k): bilinear resize, /255, (x-0.5)/0.5."""
    im = pil_img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(im).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1)


@functools.lru_cache(maxsize=1)
def legacy_lut() -> np.ndarray:
    """The composed pointwise map, as a 256-entry uint8 table (identical per channel)."""
    ramp = np.zeros((1, 256, 3), dtype=np.uint8)
    for c in range(3):
        ramp[0, :, c] = np.arange(256)
    out = np.array(_round_trip(_vit_pixel_values_from_array(ramp)))
    return out[0, :, 0].astype(np.uint8)


def _vit_pixel_values_from_array(arr_uint8: np.ndarray) -> torch.Tensor:
    arr = arr_uint8.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1)


def legacy_transform_pil(pil_img: Image.Image, alpha: float = 0.2, beta: float = 0.8,
                         size: int = 224) -> np.ndarray:
    """Byte-exact reproduction of feature_extractor_folder.py.

    Returns uint8 HWC in [0, 255]. `alpha` weights the LUT branch, `beta` the
    original -- matching the historical call site's positional argument order.

    The historical code called `img.resize((224, 224))` with no `resample`
    argument. Pillow resolves that default to BICUBIC (verified on 10.2.0), but a
    *default* is a version-dependent thing to rely on, so it is stated
    explicitly here. Byte-exactness against the surviving
    features_extracted_dataset-(0.2-0.8) was re-verified after this change.
    """
    original = np.array(pil_img.resize((size, size), Image.BICUBIC))
    lut_branch = np.array(_round_trip(_vit_pixel_values(pil_img, size)))
    return (alpha * lut_branch + beta * original).astype("uint8")   # truncation, as in the original


def legacy_transform_tensor(x: torch.Tensor, alpha: float = 0.2, beta: float = 0.8) -> torch.Tensor:
    """Table-driven equivalent for batched tensors, for use inside a dataloader.

    Input/Output: [B, 3, H, W] float in [0, 1].

    Applies the recovered LUT pointwise and blends. This agrees with
    `legacy_transform_pil` up to the resize step, which the dataloader has
    already performed.
    """
    table = torch.from_numpy(legacy_lut().astype(np.float32) / 255.0).to(x.device)
    idx = torch.clamp((x * 255.0).round(), 0, 255).long()
    return alpha * table[idx] + beta * x
