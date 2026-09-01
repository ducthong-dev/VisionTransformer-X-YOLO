#!/usr/bin/env python
"""Freeze the Controlled Synthetic Corruption Benchmark.

    python scripts/generate_controlled_corruptions.py                 # freeze the spec
    python scripts/generate_controlled_corruptions.py --hashes        # + pixel integrity
    python scripts/generate_controlled_corruptions.py --verify        # re-check the hashes

The benchmark is applied ON THE FLY at evaluation time rather than written to disk:
6 families x 3 severities x 8335 images would be ~12 GB of PNG for no scientific gain,
because every corrupted image is a pure function of

    (global_seed, relative_path, family, severity)

via `aetfpe.seeding.derive_seed` (blake2b). This script therefore freezes the
*specification* -- the exact family, severity, parameters and derived seed for every
one of the 150,030 corrupted samples -- so the benchmark is fully reproducible and
auditable without storing the images.

With `--hashes` it additionally records `sha256_array` of each corrupted image, which
is the integrity field of record: invariant to image-encoder version, so a later
regeneration on another machine either matches exactly or fails loudly. This pass is
model-independent and must be run BEFORE any model's test performance is inspected.

Reads the CLEAN TEST SPLIT ONLY. Never writes into the dataset. No model is loaded.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.corruptions import NEEDS_DISTRACTOR, apply_corruption  # noqa: E402
from aetfpe.data import list_classes, list_samples  # noqa: E402
from aetfpe.seeding import derive_seed, sha256_array, sha256_file  # noqa: E402

MANIFEST_FIELDS = ["relative_path", "class", "family", "severity", "parameters",
                   "derived_seed", "pixel_sha256"]


def load_spec(path: str) -> dict:
    import yaml
    cfg = yaml.safe_load(open(path))
    for f in cfg["corruptions"]:
        if f in NEEDS_DISTRACTOR:
            raise SystemExit(f"ERROR: family {f!r} needs a distractor image from another "
                             "class; it is not label-preserving and must not be in this benchmark.")
    return cfg


def plan(cfg: dict) -> list[tuple[str, str, dict]]:
    out = []
    for family, sevs in cfg["corruptions"].items():
        for sev, params in sevs.items():
            out.append((family, str(sev), dict(params)))
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def resize_clean(path: str, size: int) -> np.ndarray:
    """Exactly what LeafDataset feeds the model, before ToTensor: Resize((s,s)) on RGB."""
    from torchvision import transforms
    img = Image.open(path).convert("RGB")
    return np.array(transforms.Resize((size, size))(img))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/controlled_corruptions.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="results/controlled_corruptions")
    ap.add_argument("--hashes", action="store_true", help="compute pixel_sha256 (slow, model-independent)")
    ap.add_argument("--verify", action="store_true", help="recompute hashes and compare to the manifest")
    ap.add_argument("--limit-per-class", type=int, default=None, help="smoke mode only")
    args = ap.parse_args()

    if args.data_root is None:
        import yaml
        args.data_root = yaml.safe_load(open("configs/local.yaml"))["env"]["DATA_ROOT"]

    cfg = load_spec(args.spec)
    size = int(cfg["image_size"])
    seed = int(cfg["global_seed"])
    root = os.path.join(args.data_root, cfg["source_split"])
    if "test" not in os.path.basename(root):
        raise SystemExit(f"ERROR: source split must be the clean test split, got {root}")

    classes = list_classes(root)
    samples = list_samples(root, classes)
    if args.limit_per_class:
        seen: dict[int, int] = {}
        keep = []
        for s in samples:
            if seen.get(s[1], 0) < args.limit_per_class:
                keep.append(s)
                seen[s[1]] = seen.get(s[1], 0) + 1
        samples = keep
    else:
        if len(samples) != int(cfg["expected_images"]):
            raise SystemExit(f"ERROR: {len(samples)} images, spec expects {cfg['expected_images']}")
        if len(classes) != int(cfg["expected_classes"]):
            raise SystemExit(f"ERROR: {len(classes)} classes, spec expects {cfg['expected_classes']}")

    steps = plan(cfg)
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "controlled_corruption_manifest.csv.gz")

    print(f"spec        : {args.spec}  (sha256 {sha256_file(args.spec)[:16]}…)")
    print(f"source      : {root}")
    print(f"images      : {len(samples)}   families x severities: {len(steps)}")
    print(f"total rows  : {len(samples) * len(steps)}")
    print(f"pixel hashes: {'YES' if (args.hashes or args.verify) else 'no (spec-only freeze)'}\n")

    prior = {}
    if args.verify:
        if not os.path.exists(manifest_path):
            raise SystemExit(f"ERROR: {manifest_path} not found; nothing to verify.")
        with gzip.open(manifest_path, "rt") as fh:
            for r in csv.DictReader(fh):
                prior[(r["relative_path"], r["family"], r["severity"])] = r["pixel_sha256"]

    rows = []
    mismatches = []
    t0 = time.time()
    for i, (path, label) in enumerate(samples):
        rel = os.path.relpath(path, root)
        base = resize_clean(path, size) if (args.hashes or args.verify) else None
        for family, sev, params in steps:
            ds = derive_seed(rel, family, sev, base=seed)
            px = ""
            if base is not None:
                out = apply_corruption(base, family, params, rel, sev, seed=seed)
                px = sha256_array(out)
                if args.verify:
                    was = prior.get((rel, family, sev))
                    if was and was != px:
                        mismatches.append(f"{rel} {family}/{sev}: manifest {was[:12]}… now {px[:12]}…")
            rows.append({"relative_path": rel, "class": classes[label], "family": family,
                         "severity": sev, "parameters": json.dumps(params, sort_keys=True),
                         "derived_seed": ds, "pixel_sha256": px})
        if (i + 1) % 250 == 0 or i + 1 == len(samples):
            el = time.time() - t0
            eta = el / (i + 1) * (len(samples) - i - 1)
            print(f"  {i+1}/{len(samples)} images  {el:6.1f}s elapsed  ~{eta/60:5.1f} min left", flush=True)

    if args.verify:
        print(f"\nverified {len(rows)} rows; {len(mismatches)} mismatches")
        for m in mismatches[:10]:
            print("  MISMATCH", m)
        return 1 if mismatches else 0

    tmp = manifest_path + ".tmp"
    with gzip.open(tmp, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, manifest_path)

    spec_record = {
        "name": cfg["name"],
        "version": cfg["version"],
        "frozen": cfg["frozen"],
        "not_pre_registered": ("Written after training completed, before any test performance "
                               "was inspected. This is a frozen specification, NOT a pre-registration."),
        "introduced_because": ("The corruption-construction audit (docs/CORRUPTION_PROTOCOL.md) showed "
                               "Easy/Moderate/Hard mix photometric corruption with geometric "
                               "augmentation (~84% of `hard` flipped or rotated), so degradation on "
                               "them cannot be attributed to noise robustness."),
        "relationship_to_existing": ("ADDITIONAL, not a replacement. Clean/Easy/Moderate/Hard remain "
                                     "unchanged and are reported separately. The two benchmarks are "
                                     "never merged into a single average."),
        "spec_path": args.spec,
        "spec_sha256": sha256_file(args.spec),
        "git_commit": git_commit(),
        "global_seed": seed,
        "seed_rule": cfg["seed_rule"],
        "excluded_transform_classes": cfg["excluded_transform_classes"],
        "families": {f: list(s) for f, s in cfg["corruptions"].items()},
        "display_names": cfg["display_names"],
        "source_split": cfg["source_split"],
        "source_root": root,
        "num_images": len(samples),
        "num_distributions": len(steps),
        "manifest": os.path.basename(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_rows": len(rows),
        "pixel_hashes_present": bool(args.hashes),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pillow": Image.__version__},
    }
    with open(os.path.join(args.out, "controlled_corruption_spec.json"), "w") as fh:
        json.dump(spec_record, fh, indent=2)

    print(f"\nwrote {manifest_path}")
    print(f"      {os.path.join(args.out, 'controlled_corruption_spec.json')}")
    print(f"manifest sha256 {spec_record['manifest_sha256'][:16]}…  commit {spec_record['git_commit'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
