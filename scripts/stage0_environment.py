#!/usr/bin/env python
"""Stage 0 — environment verification (COLAB_CAMPAIGN_PLAN.md).

Captures every field the frozen protocol requires before any data or model work
begins, and enforces the one hard rule for this stage: **a dirty repository
fails Stage 0.** If the working tree does not match the committed
`revision-protocol-v1` tag exactly, nothing downstream can be trusted to have
run under the frozen protocol.

    COLAB   python scripts/stage0_environment.py --require-cuda
    LOCAL   python scripts/stage0_environment.py            # dev dry run, CUDA not required

Writes ${OUTPUT_ROOT}/environment/stage0_environment.json and exits non-zero on
any failure (dirty repo, or --require-cuda with no CUDA device).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np
import PIL
import PIL.features
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import git_commit, git_dirty, resolve_roots  # noqa: E402


def codec_version(name: str):
    try:
        return PIL.features.version_codec(name)
    except Exception:  # noqa: BLE001
        return None


def cuda_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version_torch_built_with": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
    }


def ultralytics_version():
    try:
        import ultralytics

        return ultralytics.__version__
    except ImportError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-cuda", action="store_true",
                    help="fail if no CUDA device is present (use on Colab)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="override the dirty-repo failure (development only; "
                         "never use for an official Stage 0 run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    args.out = args.out or os.path.join(resolve_roots()["OUTPUT_ROOT"], "environment",
                                        "stage0_environment.json")

    dirty = git_dirty()
    commit = git_commit()
    cuda = cuda_info()
    zlib_v = __import__("zlib").ZLIB_RUNTIME_VERSION

    report = {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_built_with_cuda": torch.backends.cuda.is_built(),
        "cuda": cuda,
        "ultralytics": ultralytics_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "codec_jpeg": codec_version("jpg"),
        "codec_zlib": codec_version("zlib"),
        "zlib_runtime": zlib_v,
    }

    failures = []
    if dirty and not args.allow_dirty:
        failures.append(f"repository is dirty at commit {commit[:12]} -- "
                        f"Stage 0 requires an exact match to the frozen tag")
    if args.require_cuda and not cuda["available"]:
        failures.append("CUDA not available, but --require-cuda was set")
    if report["pillow"] != "10.2.0":
        failures.append(f"Pillow is {report['pillow']}, expected 10.2.0 (pinned in "
                        f"requirements.txt for JPEG-corruption reproducibility)")

    report["stage0_pass"] = not failures
    report["failures"] = failures

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")
    if failures:
        print("\nSTAGE 0: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nSTAGE 0: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
