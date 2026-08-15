#!/usr/bin/env python
"""Stage 1 -- dataset verification (COLAB_CAMPAIGN_PLAN.md).

Confirms the mounted dataset is the correct one before anything touches it. Two
copies of "the PlantVillage dataset" exist across this project's history and
they are NOT interchangeable:

    correct copy   38,584 / 8,340 / 8,335   train/val/test, 39 classes
    sibling copy    38,584 / 8,346 / 8,334   (FPT/YOLOv8-ResCBAM's dataset.yaml)

Pointing DATA_ROOT at the wrong one silently shifts 6 images between val and
test and breaks comparability with every historical number
(IMPLEMENTATION_VALIDATION.md Section 2.1).

    python scripts/verify_dataset.py

Writes:
  ${OUTPUT_ROOT}/dataset/dataset_manifest.csv   one row per image
  ${OUTPUT_ROOT}/dataset/dataset_summary.json   counts, hash, pass/fail

Exits non-zero -- STOP -- if any split's count or the class count differs from
the frozen expectation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import environment_info, load_experiment, resolve_roots  # noqa: E402
from aetfpe.data import IMG_EXT, list_classes  # noqa: E402

# RECOVERED: log-org-280223 + the on-disk dataset this framework targets.
EXPECTED = {"train": 38584, "val": 8340, "test": 8335, "num_classes": 39}
TOLERANCE = 0  # exact match required; this is not a fuzzy check


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/_base.yaml")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    d = cfg["data"]
    root = d["root"]
    out_dir = args.out_dir or os.path.join(resolve_roots()["OUTPUT_ROOT"], "dataset")
    os.makedirs(out_dir, exist_ok=True)

    train_root = os.path.join(root, d["train_split"])
    classes = list_classes(train_root)

    manifest_path = os.path.join(out_dir, "dataset_manifest.csv")
    rows = []
    counts = {}
    per_class = {"train": {}, "val": {}, "test": {}}

    for split in ("train", "val", "test"):
        split_root = os.path.join(root, d[f"{split}_split"])
        n = 0
        for c in classes:
            cd = os.path.join(split_root, c)
            files = sorted(f for f in os.listdir(cd) if f.endswith(IMG_EXT)) if os.path.isdir(cd) else []
            per_class[split][c] = len(files)
            n += len(files)
            for f in files:
                rows.append({"split": split, "class": c, "filename": f,
                            "relative_path": os.path.relpath(os.path.join(cd, f), root)})
        counts[split] = n

    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "class", "filename", "relative_path"])
        w.writeheader()
        w.writerows(rows)

    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: (r["split"], r["class"], r["filename"])):
        h.update(f"{r['split']}|{r['class']}|{r['filename']}".encode())
    manifest_hash = h.hexdigest()

    checks = {
        "num_classes": (len(classes), EXPECTED["num_classes"]),
        "train": (counts["train"], EXPECTED["train"]),
        "val": (counts["val"], EXPECTED["val"]),
        "test": (counts["test"], EXPECTED["test"]),
    }
    mismatches = {k: v for k, v in checks.items() if abs(v[0] - v[1]) > TOLERANCE}

    summary = {
        "data_root": root,
        "num_classes": len(classes),
        "classes": classes,
        "counts": counts,
        "expected": EXPECTED,
        "mismatches": {k: {"got": v[0], "expected": v[1]} for k, v in mismatches.items()},
        "manifest_path": manifest_path,
        "manifest_rows": len(rows),
        "manifest_sha256": manifest_hash,
        "environment": environment_info(),
        "pass": not mismatches,
    }
    with open(os.path.join(out_dir, "dataset_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"data root : {root}")
    print(f"classes   : {len(classes)} (expected {EXPECTED['num_classes']})")
    for split in ("train", "val", "test"):
        print(f"{split:6s}: {counts[split]:6d} images (expected {EXPECTED[split]})")
    print(f"\nmanifest sha256: {manifest_hash}")
    print(f"manifest       : {manifest_path} ({len(rows)} rows)")

    if mismatches:
        print("\nSTOP -- dataset does not match the frozen expectation:")
        for k, (got, exp) in mismatches.items():
            print(f"  {k}: got {got}, expected {exp}")
        print("\nThis usually means the wrong dataset copy is mounted -- see the "
              "docstring above. Do NOT proceed to Stage 2.")
        return 1

    print("\nSTAGE 1: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
