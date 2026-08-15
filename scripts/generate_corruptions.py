#!/usr/bin/env python
"""Generate the frozen corrupted test sets, once, with a manifest.

Every model is evaluated on these exact files. Regenerating with the same seed
reproduces byte-identical output, which `--verify` checks against the recorded
sha256 without needing a second copy on disk.

    python scripts/generate_corruptions.py --out data/corruptions
    python scripts/generate_corruptions.py --out data/corruptions --limit-per-class 2
    python scripts/generate_corruptions.py --out data/corruptions --verify

Only the TEST split is ever corrupted. Training and validation stay clean unless
an experiment explicitly studies corruption-aware training (arm M3), which
applies its augmentation in the dataloader and never touches these files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import load_yaml, resolve_roots  # noqa: E402
from aetfpe.corruptions import NEEDS_DISTRACTOR, apply_corruption, expand_plan  # noqa: E402
from aetfpe.data import IMG_EXT, list_classes  # noqa: E402
from aetfpe.seeding import (  # noqa: E402
    derive_seed, generation_environment, seed_everything, sha256_array, sha256_file,
)

# `pixel_sha256` is the integrity field of record: it hashes the decoded pixel
# array, so it is invariant to PNG encoder version and settings. `file_sha256`
# hashes the encoded bytes and is informational only -- zlib version and encoder
# flags change it without changing a single pixel.
MANIFEST_FIELDS = [
    "original_path", "corrupted_path", "class", "corruption",
    "severity", "seed", "parameters", "pixel_sha256", "file_sha256",
]


def collect_samples(test_root: str, classes: list[str], limit_per_class: int | None):
    samples = []
    for c in classes:
        d = os.path.join(test_root, c)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(IMG_EXT))
        if limit_per_class:
            files = files[:limit_per_class]
        for f in files:
            samples.append((c, f, os.path.join(d, f)))
    return samples


def pick_distractor(samples, index: int, cls: str, rel_path: str, corruption: str,
                    severity: str, seed: int) -> str:
    """Deterministically choose an image from a DIFFERENT class."""
    rng = np.random.default_rng(derive_seed(rel_path, corruption, severity, "distractor", base=seed))
    for _ in range(64):
        j = int(rng.integers(0, len(samples)))
        if samples[j][0] != cls:
            return samples[j][2]
    # degenerate fallback: first sample of any other class
    for c2, _, p2 in samples:
        if c2 != cls:
            return p2
    raise RuntimeError("dataset has only one class; transparency corruption is undefined")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/corruptions.yaml")
    ap.add_argument("--data-config", default="configs/_base.yaml")
    ap.add_argument("--out", default=None, help="default: ${OUTPUT_ROOT}/corruptions")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="smoke-test mode: only this many images per class")
    ap.add_argument("--split", default=None,
                    help="which split to corrupt. 'test' is the frozen final benchmark; "
                         "'val' produces the calibration set used for any hyperparameter "
                         "decision, so the test set is never observed during development.")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these corruption names")
    ap.add_argument("--verify", action="store_true",
                    help="regenerate in memory and compare against the manifest checksums")
    args = ap.parse_args()

    ccfg = load_yaml(args.config)
    base = load_yaml(args.data_config)
    seed = int(ccfg.get("seed", 0))
    size = int(ccfg.get("image_size", 224))
    split = args.split or ccfg.get("split", "test")
    # test -> results/corruptions (the frozen benchmark)
    # val  -> results/corruptions_val (the calibration set)
    default_name = "corruptions" if split == "test" else f"corruptions_{split}"
    args.out = args.out or os.path.join(resolve_roots()["OUTPUT_ROOT"], default_name)
    seed_everything(seed)

    data_root = base["data"]["root"]
    test_root = os.path.join(data_root, base["data"].get(f"{split}_split", split))
    train_root = os.path.join(data_root, base["data"]["train_split"])

    classes = list_classes(train_root)              # canonical class list
    samples = collect_samples(test_root, classes, args.limit_per_class)
    plan = expand_plan(ccfg)
    if args.only:
        plan = [p for p in plan if p[0] in set(args.only) or p[0] == "clean"]

    print(f"source split : {test_root}")
    print(f"classes      : {len(classes)}")
    print(f"images       : {len(samples)}"
          + (f"  (limited to {args.limit_per_class}/class)" if args.limit_per_class else ""))
    print(f"corruptions  : {len(plan)} configurations")
    print(f"output root  : {args.out}")
    print(f"total files  : {len(samples) * len(plan)}\n")

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "corruption_manifest.csv")

    if args.verify:
        return verify(manifest_path, samples, plan, seed, size, args.out)

    rows = []
    for corruption, severity, params in plan:
        sub = "clean" if corruption == "clean" else os.path.join(corruption, severity)
        for i, (cls, fname, src) in enumerate(samples):
            rel = os.path.join(cls, fname)
            img = np.array(Image.open(src).convert("RGB").resize((size, size), Image.BICUBIC))

            distractor = None
            if corruption in NEEDS_DISTRACTOR:
                dp = pick_distractor(samples, i, cls, rel, corruption, severity, seed)
                distractor = np.array(
                    Image.open(dp).convert("RGB").resize((size, size), Image.BICUBIC)
                )

            out_arr = apply_corruption(img, corruption, params, rel, severity, seed, distractor)

            dst_dir = os.path.join(args.out, sub, cls)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.splitext(fname)[0] + ".png")
            Image.fromarray(out_arr).save(dst, format="PNG", optimize=False)

            rows.append({
                "original_path": os.path.relpath(src, data_root),
                "corrupted_path": os.path.relpath(dst, args.out),
                "class": cls,
                "corruption": corruption,
                "severity": severity,
                "seed": seed,
                "parameters": json.dumps(params, sort_keys=True),
                "pixel_sha256": sha256_array(out_arr),
                "file_sha256": sha256_file(dst),
            })
        print(f"  done {corruption}/{severity}: {len(samples)} images")

    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    bundle_path = write_persistence_bundle(
        args.out, ccfg, base, split, classes, samples, plan, seed, size,
        len(rows), manifest_path,
    )

    print(f"\nmanifest : {manifest_path}  ({len(rows)} rows)")
    print(f"manifest sha256 : {sha256_file(manifest_path)}")
    print(f"bundle   : {bundle_path}")
    print("\nPERSISTENT artifacts (archive these; a few MB):")
    print(f"  {os.path.basename(manifest_path)}")
    print("  clean_split_manifest.csv")
    print("  generation_environment.json")
    print("  configs/corruptions.yaml  (hash recorded in the bundle)")
    print("EPHEMERAL artifacts (regenerable; do not archive):")
    print(f"  {len(rows)} PNG files under {args.out}")
    return 0


def verify(manifest_path, samples, plan, seed, size, out_root) -> int:
    """Regenerate every corrupted image in memory and compare hashes.

    PIXEL hashes are the pass/fail criterion: they are invariant to image-encoder
    version and settings, so they are what must hold across macOS and Colab.
    ENCODED-file hashes are also compared, but a file-only mismatch is reported
    as a warning -- it means the PNG encoder changed, not the benchmark.
    """
    import hashlib
    import io

    if not os.path.exists(manifest_path):
        print(f"no manifest at {manifest_path}; run without --verify first")
        return 1

    by_key = {}
    with open(manifest_path) as fh:
        for r in csv.DictReader(fh):
            key = (r["corruption"], r["severity"], r["corrupted_path"])
            by_key[key] = (r.get("pixel_sha256"), r.get("file_sha256") or r.get("checksum"))

    checked = pixel_bad = file_bad = 0
    for corruption, severity, params in plan:
        sub = "clean" if corruption == "clean" else os.path.join(corruption, severity)
        for i, (cls, fname, src) in enumerate(samples):
            rel = os.path.join(cls, fname)
            key = (corruption, severity,
                   os.path.join(sub, cls, os.path.splitext(fname)[0] + ".png"))
            if key not in by_key:
                continue
            want_pixel, want_file = by_key[key]

            img = np.array(Image.open(src).convert("RGB").resize((size, size), Image.BICUBIC))
            distractor = None
            if corruption in NEEDS_DISTRACTOR:
                dp = pick_distractor(samples, i, cls, rel, corruption, severity, seed)
                distractor = np.array(
                    Image.open(dp).convert("RGB").resize((size, size), Image.BICUBIC)
                )
            arr = apply_corruption(img, corruption, params, rel, severity, seed, distractor)

            checked += 1
            if want_pixel and sha256_array(arr) != want_pixel:
                pixel_bad += 1
                if pixel_bad <= 5:
                    print(f"  PIXEL MISMATCH {key[2]}")

            if want_file:
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="PNG", optimize=False)
                if hashlib.sha256(buf.getvalue()).hexdigest() != want_file:
                    file_bad += 1

    print(f"\nverified {checked} images")
    print(f"  pixel-content mismatches : {pixel_bad}")
    print(f"  encoded-file mismatches  : {file_bad}")
    if pixel_bad == 0 and file_bad > 0:
        print("\n  NOTE: pixels are identical but PNG bytes differ. The image encoder "
              "(Pillow/zlib) changed.\n  The benchmark is intact -- regenerate the files "
              "and continue.")
    if pixel_bad:
        print("\n  FAIL: pixel content changed. Do NOT use this regeneration. Check "
              "generation_environment.json\n  against the current environment "
              "(numpy, Pillow, and especially the JPEG codec).")
    return 0 if pixel_bad == 0 else 2


def write_persistence_bundle(out_dir: str, ccfg: dict, base: dict, split: str,
                             classes: list[str], samples: list, plan: list, seed: int,
                             size: int, n_rows: int, manifest_path: str) -> str:
    """Write the small, permanently-archivable record of this generation.

    The corrupted PNGs are ephemeral: they are ~21 GB and regenerable. What must
    survive is this bundle, which is a few MB and is sufficient to reproduce and
    verify the benchmark from scratch.
    """
    from aetfpe.config import environment_info, git_commit, git_dirty

    split_manifest = os.path.join(out_dir, "clean_split_manifest.csv")
    with open(split_manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "class", "filename", "source_path"])
        for cls, fname, src in samples:
            w.writerow([split, cls, fname, src])

    bundle = {
        "generator_version": "aetfpe.corruptions/1.0",
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "seed": seed,
        "image_size": size,
        "resample": "PIL BICUBIC (explicit, not the version-dependent default)",
        "split": split,
        "num_classes": len(classes),
        "num_source_images": len(samples),
        "num_configurations": len(plan),
        "num_generated_files": n_rows,
        "configurations": [{"corruption": c, "severity": s, "parameters": p}
                           for c, s, p in plan],
        "corruption_config_sha256": sha256_file(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "configs", "corruptions.yaml")),
        "manifest_sha256": sha256_file(manifest_path),
        "clean_split_manifest_sha256": sha256_file(split_manifest),
        "generation_environment": generation_environment(),
        "runtime_environment": environment_info(),
        "integrity_policy": {
            "field_of_record": "pixel_sha256",
            "informational": "file_sha256",
            "rationale": "PNG bytes depend on zlib/encoder version; pixel content does not.",
        },
    }
    path = os.path.join(out_dir, "generation_environment.json")
    with open(path, "w") as fh:
        json.dump(bundle, fh, indent=2, default=str)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
