"""Broader robustness corruptions requested by Reviewer #11.

All operate on uint8 HWC arrays and take an explicit numpy Generator, so every
corrupted image is reproducible from (global seed, relative path, corruption,
severity) alone. Severity parameters live in configs/corruptions.yaml and were
fixed *before* any model was evaluated.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise. `sigma` in 0-255 units."""
    noise = rng.normal(0.0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    """Normalised 1-D Gaussian, truncated at 3 sigma. float64 throughout."""
    radius = int(np.ceil(3.0 * float(sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * float(sigma) ** 2))
    return k / k.sum()


def gaussian_blur(img: np.ndarray, sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Separable isotropic Gaussian blur, implemented in numpy.

    Deliberately NOT `PIL.ImageFilter.GaussianBlur`. Pillow's blur implementation
    has changed between releases, which would make this corruption's *pixel
    content* depend on the installed Pillow version. Everything here is numpy
    float64 with an explicit kernel and edge padding, so the output is identical
    on any platform with a conforming numpy.

    `rng` is accepted for signature uniformity and not consumed.
    """
    k = _gaussian_kernel1d(sigma)
    r = len(k) // 2
    x = img.astype(np.float64)
    x = np.pad(x, ((r, r), (r, r), (0, 0)), mode="edge")

    # horizontal then vertical pass
    acc = np.zeros((x.shape[0], img.shape[1], img.shape[2]), dtype=np.float64)
    for i, w in enumerate(k):
        acc += w * x[:, i : i + img.shape[1], :]
    out = np.zeros(img.shape, dtype=np.float64)
    for i, w in enumerate(k):
        out += w * acc[i : i + img.shape[0], :, :]

    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def brightness(img: np.ndarray, factor: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Multiplicative brightness change. factor < 1 darkens. Deterministic."""
    return np.clip(img.astype(np.float32) * float(factor), 0, 255).astype(np.uint8)


def contrast(img: np.ndarray, factor: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Contrast scaling about the per-image mean. Deterministic."""
    mean = img.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    return np.clip((img.astype(np.float32) - mean) * float(factor) + mean, 0, 255).astype(np.uint8)


def jpeg_compression(img: np.ndarray, quality: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Round-trip through JPEG at the given quality.

    Deterministic within one environment, but **not** guaranteed bit-identical
    across libjpeg implementations: libjpeg-turbo and reference libjpeg differ,
    and turbo releases have changed their DCT/quantisation paths. This is the one
    corruption whose *pixel array* can vary between environments.

    Mitigation, rather than pretence: the generation environment records the JPEG
    codec version in `generation_environment.json`, and the manifest stores a
    pixel-content hash. A regeneration on a different libjpeg will therefore fail
    verification loudly instead of silently changing the benchmark.
    """
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def motion_blur(img: np.ndarray, length: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Horizontal box-average motion blur of `length` pixels. Deterministic."""
    k = max(int(length), 1)
    if k == 1:
        return img.copy()
    pad = k // 2
    padded = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode="edge").astype(np.float32)
    acc = np.zeros_like(img, dtype=np.float32)
    for i in range(k):
        acc += padded[:, i : i + img.shape[1], :]
    return np.clip(acc / k, 0, 255).astype(np.uint8)
