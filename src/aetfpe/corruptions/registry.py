"""Corruption registry + the deterministic apply() entry point.

Everything the generator needs to turn one clean image into one corrupted image
lives here, so the generator script itself stays thin and the behaviour is
testable in isolation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..seeding import image_rng
from . import common, legacy

# name -> (callable, parameter name, needs a distractor image?)
REGISTRY: dict[str, tuple[Callable, str, bool]] = {
    "pepper": (legacy.salt_and_pepper, "ratio", False),
    "transparency": (legacy.transparency_overlay, "alpha", True),
    "pepper_transparency": (legacy.pepper_then_transparency, "ratio", True),
    "gaussian_noise": (common.gaussian_noise, "sigma", False),
    "gaussian_blur": (common.gaussian_blur, "sigma", False),
    "brightness": (common.brightness, "factor", False),
    "contrast": (common.contrast, "factor", False),
    "jpeg": (common.jpeg_compression, "quality", False),
    "motion_blur": (common.motion_blur, "length", False),
}

NEEDS_DISTRACTOR = {k for k, (_, _, d) in REGISTRY.items() if d}


def apply_corruption(
    img: np.ndarray,
    corruption: str,
    params: dict,
    rel_path: str,
    severity: str,
    seed: int = 0,
    distractor: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministically corrupt one image.

    The RNG is derived from (seed, rel_path, corruption, severity), so the result
    does not depend on iteration order, worker count, or how many images were
    processed before it.
    """
    if corruption == "clean":
        return img.copy()
    if corruption not in REGISTRY:
        raise KeyError(f"unknown corruption {corruption!r}; available: {sorted(REGISTRY)}")

    fn, _, needs_distractor = REGISTRY[corruption]
    rng = image_rng(rel_path, corruption, severity, base=seed)

    if needs_distractor:
        if distractor is None:
            raise ValueError(f"corruption {corruption!r} requires a distractor image")
        if corruption == "transparency":
            return fn(img, distractor, alpha=float(params["alpha"]))
        return fn(
            img,
            distractor,
            ratio=float(params["ratio"]),
            rng=rng,
            alpha=float(params.get("alpha", 0.7)),
            salt_vs_pepper=float(params.get("salt_vs_pepper", 0.5)),
        )

    if corruption == "pepper":
        return fn(
            img,
            ratio=float(params["ratio"]),
            rng=rng,
            salt_vs_pepper=float(params.get("salt_vs_pepper", 0.5)),
            per_channel=bool(params.get("per_channel", False)),
        )

    key = REGISTRY[corruption][1]
    return fn(img, params[key], rng)


def expand_plan(cfg: dict) -> list[tuple[str, str, dict]]:
    """configs/corruptions.yaml -> flat list of (corruption, severity, params)."""
    plan: list[tuple[str, str, dict]] = [("clean", "none", {})]
    for corruption, severities in cfg.get("corruptions", {}).items():
        for severity, params in severities.items():
            plan.append((corruption, str(severity), dict(params)))
    return plan
