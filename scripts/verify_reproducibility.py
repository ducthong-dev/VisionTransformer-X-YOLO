#!/usr/bin/env python
"""Cross-platform reproducibility probe for the corruption pipeline.

Answers, empirically rather than by assertion: does this environment produce the
same PIXELS as the reference environment?

It runs every corruption on a fixed synthetic image derived from a fixed seed --
no dataset needed -- and hashes the resulting pixel arrays. Run it on the
development machine to write the reference, then on Colab to compare.

    # reference machine (once)
    python scripts/verify_reproducibility.py --write-reference

    # Colab, before Stage 2
    python scripts/verify_reproducibility.py --check

Exit code 0 = all families match, 2 = at least one family diverges.

The distinction this script exists to make:

  PIXEL-ARRAY reproducibility   the corruption arithmetic gives the same numbers.
                                This is what the benchmark requires.
  ENCODED-FILE reproducibility  the PNG/JPEG bytes are identical. This depends on
                                zlib and libjpeg builds and is NOT required, and
                                NOT guaranteed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import load_yaml  # noqa: E402
from aetfpe.corruptions import NEEDS_DISTRACTOR, apply_corruption, expand_plan  # noqa: E402
from aetfpe.seeding import generation_environment, sha256_array  # noqa: E402

REFERENCE = "docs/reproducibility_reference.json"

# How each family's pixel output depends on the environment.
DEPENDENCY = {
    "clean": "numpy only",
    "pepper": "numpy only",
    "transparency": "numpy only",
    "pepper_transparency": "numpy only",
    "gaussian_noise": "numpy only",
    "gaussian_blur": "numpy only",
    "brightness": "numpy only",
    "contrast": "numpy only",
    "motion_blur": "numpy only",
    "jpeg": "LIBJPEG-DEPENDENT",
}


def synthetic_image(seed: int = 0, size: int = 224) -> np.ndarray:
    """A fixed, dataset-independent test image. Structure + gradient + noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = np.stack([
        (xx * 255 // size),
        (yy * 255 // size),
        ((xx + yy) * 255 // (2 * size)),
    ], axis=-1).astype(np.float64)
    base += rng.integers(0, 40, size=(size, size, 3))
    base[size // 4 : size // 2, size // 4 : size // 2, :] = 250   # a flat patch
    return np.clip(base, 0, 255).astype(np.uint8)


def probe(config: str) -> dict:
    ccfg = load_yaml(config)
    size = int(ccfg.get("image_size", 224))
    seed = int(ccfg.get("seed", 0))
    img = synthetic_image(0, size)
    distractor = synthetic_image(1, size)

    results = {}
    for corruption, severity, params in expand_plan(ccfg):
        arr = apply_corruption(
            img, corruption, params, "probe/synthetic.png", severity, seed,
            distractor if corruption in NEEDS_DISTRACTOR else None,
        )
        results[f"{corruption}/{severity}"] = {
            "pixel_sha256": sha256_array(arr),
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "dependency": DEPENDENCY.get(corruption, "unknown"),
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/corruptions.yaml")
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--write-reference", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    env = generation_environment()
    results = probe(args.config)

    if args.write_reference or not args.check:
        os.makedirs(os.path.dirname(args.reference) or ".", exist_ok=True)
        payload = {"environment": env, "image_probe": "synthetic, seed 0", "results": results}
        if args.write_reference:
            with open(args.reference, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"wrote reference for {len(results)} configurations -> {args.reference}")
            print(f"environment: numpy {env['numpy']}, Pillow {env['pillow']}, "
                  f"jpeg {env['codec_jpeg']}, {env['machine']}")
        else:
            print(json.dumps(payload, indent=2))
        return 0

    if not os.path.exists(args.reference):
        print(f"no reference at {args.reference}; run --write-reference on the "
              f"reference machine first")
        return 1

    ref = json.load(open(args.reference))
    print("reference env :", ref["environment"]["machine"],
          f"numpy {ref['environment']['numpy']}",
          f"Pillow {ref['environment']['pillow']}",
          f"jpeg {ref['environment']['codec_jpeg']}")
    print("this env      :", env["machine"], f"numpy {env['numpy']}",
          f"Pillow {env['pillow']}", f"jpeg {env['codec_jpeg']}")
    print()

    bad, missing = [], []
    for key, val in results.items():
        if key not in ref["results"]:
            missing.append(key)
            continue
        if val["pixel_sha256"] != ref["results"][key]["pixel_sha256"]:
            bad.append((key, val["dependency"], ref["results"][key]["mean"], val["mean"]))

    for key, dep, want, got in bad:
        print(f"  DIVERGES  {key:34s} [{dep}]  mean {want} -> {got}")
    for key in missing:
        print(f"  MISSING   {key}")

    print(f"\n{len(results) - len(bad) - len(missing)}/{len(results)} configurations "
          f"reproduce byte-identically at the pixel level")

    if bad:
        codec_only = all(d == "LIBJPEG-DEPENDENT" for _, d, _, _ in bad)
        if codec_only:
            print("\n  All divergences are in the JPEG family, whose pixel output depends on\n"
                  "  the libjpeg build. Options: (a) pin Pillow to the reference version, or\n"
                  "  (b) drop the jpeg family from the benchmark. Do NOT proceed with a\n"
                  "  partially-divergent benchmark.")
        else:
            print("\n  Divergence outside the JPEG family means numpy arithmetic differs.\n"
                  "  Stop and reconcile numpy versions before generating anything.")
    return 0 if not bad and not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
