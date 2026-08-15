"""Legacy corruptions: the manuscript's Type 1-3 degradations.

RECONSTRUCTED. No generation code survives (docs/ARCHITECTURE_RECOVERY.md,
goal 8). What survives is only the *names* of the produced directories
(`0.02-noise` ... `0.5-noise`, `Plant_Handled-hard-*`) inside
`evaluation_results/`, and the prose in manuscript S4.3. Every parameter below
is therefore a documented reconstruction, not a recovered value.

Two contradictions in the source prose had to be resolved:

1. S4.3.1 describes Type 1 as "white and black dots" (salt-and-pepper) while
   S4.6 describes it as "randomly replaces pixels with black (pepper) values"
   (pepper only). Default here is salt-and-pepper at a 50/50 split, with
   `salt_vs_pepper=0.0` available to reproduce a pepper-only reading.

2. S4.3.2 says the blend makes "the features of the image with higher
   transparency ... more prominent", which inverts itself. Default here reads
   "transparency 70%" as: the labelled image is composited at alpha=0.7 and
   therefore dominates. The label always follows the foreground image.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- Type 1 --- #


def salt_and_pepper(
    img: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    salt_vs_pepper: float = 0.5,
    per_channel: bool = False,
) -> np.ndarray:
    """Replace `ratio` of pixels with pure black or pure white.

    img            : uint8 HWC
    ratio          : fraction of pixels replaced, in [0, 1]
    salt_vs_pepper : fraction of the replaced pixels set to white (255)
    per_channel    : if True each channel is corrupted independently; the default
                     corrupts whole pixels, which is the standard definition and
                     the one the manuscript's "dots" wording implies.
    """
    out = img.copy()
    h, w = img.shape[:2]

    if per_channel:
        mask = rng.random(img.shape) < ratio
        salt = rng.random(img.shape) < salt_vs_pepper
        out[mask & salt] = 255
        out[mask & ~salt] = 0
        return out

    mask = rng.random((h, w)) < ratio
    salt = rng.random((h, w)) < salt_vs_pepper
    out[mask & salt] = 255
    out[mask & ~salt] = 0
    return out


# ---------------------------------------------------------------- Type 2 --- #


def transparency_overlay(
    img: np.ndarray,
    other: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """I_out = alpha * I_foreground + (1 - alpha) * I_background.

    `img` is the labelled foreground; `other` is a distractor image drawn from a
    *different* class so the composite genuinely challenges the classifier. The
    caller is responsible for the deterministic choice of `other`.
    """
    if other.shape != img.shape:
        raise ValueError(f"overlay shape mismatch: {other.shape} vs {img.shape}")
    blended = alpha * img.astype(np.float32) + (1.0 - alpha) * other.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- Type 3 --- #


def pepper_then_transparency(
    img: np.ndarray,
    other: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
    alpha: float = 0.7,
    salt_vs_pepper: float = 0.5,
) -> np.ndarray:
    """S4.3.3 order: noise first, then overlay.

    The noise is applied to the labelled foreground only; the distractor stays
    clean. That is the reading that makes Type 3 a strict superset of Type 1,
    which is what the manuscript's framing ("combines the challenges introduced
    in Type 1 and Type 2") requires.
    """
    noisy = salt_and_pepper(img, ratio, rng, salt_vs_pepper=salt_vs_pepper)
    return transparency_overlay(noisy, other, alpha=alpha)
