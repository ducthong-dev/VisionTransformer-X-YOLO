"""Deterministic seeding used by every script in this project.

Every experiment records the seed it used in its result JSON. Corruption
generation additionally derives a *per-image* seed from (global_seed, relative
path, corruption, severity) so that a corrupted file is reproducible in
isolation, without depending on directory iteration order.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import torch

GLOBAL_SEED = 0


def seed_everything(seed: int = GLOBAL_SEED, deterministic: bool = True) -> None:
    """Seed python, numpy and torch. Call once at the top of every script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def derive_seed(*parts: object, base: int = GLOBAL_SEED) -> int:
    """Stable 32-bit seed derived from `base` and the string form of `parts`.

    Uses blake2b rather than hash() because python's string hashing is salted
    per process, which would silently break reproducibility across runs.
    """
    payload = "|".join([str(base)] + [str(p) for p in parts]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32)


def image_rng(path: str, corruption: str, severity: object, base: int = GLOBAL_SEED):
    """numpy Generator for one (image, corruption, severity) triple."""
    return np.random.default_rng(derive_seed(path, corruption, severity, base=base))


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Hash of the ENCODED file bytes.

    Informational only. PNG bytes depend on zlib version and encoder settings, so
    this is not a cross-environment integrity check -- use `sha256_array`.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_array(arr) -> str:
    """Hash of the raw PIXEL CONTENT: dtype, shape and C-ordered bytes.

    This is the integrity field of record. It is invariant to image-encoder
    version and settings, so it survives regeneration on a different machine as
    long as the corruption arithmetic itself is reproducible.
    """
    import numpy as np

    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def generation_environment() -> dict:
    """Everything that could change a generated pixel, captured at generation time."""
    import platform
    import zlib

    import numpy as np
    import PIL
    import PIL.features

    def codec(name):
        try:
            return PIL.features.version_codec(name)
        except Exception:  # noqa: BLE001
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "codec_jpeg": codec("jpg"),
        "codec_zlib": codec("zlib"),
        "rng": "numpy.random.Generator(PCG64), stream-stable per NumPy policy",
    }
