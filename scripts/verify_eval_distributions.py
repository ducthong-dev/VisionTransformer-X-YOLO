#!/usr/bin/env python
"""Verify the four evaluation distributions and recover the clean<->augmented mapping.

The four distributions are Clean (`test`) plus three pre-made augmented copies. They
were NOT produced by `scripts/generate_corruptions.py`; they come from the sibling
ResCBAM project's `augment_pt.py`, which saved

    <save_dir>/<class_name>/img_<GLOBAL_DATASET_INDEX>.png

where the index runs over `LeafDataset(test_dir)` -- classes sorted, files in
`os.listdir` order (NOT sorted). That makes the mapping recoverable but
filesystem-order dependent, so this script freezes it to disk once and every later
step reads the frozen manifest rather than calling `os.listdir` again.

    python scripts/verify_eval_distributions.py --out results/eval_integrity

Checks performed:
  1. structure      -- 8335 samples / 39 classes / identical class ordering / per-class supports
  2. distinctness   -- content hashes prove Easy/Moderate/Hard are not byte-identical sets
  3. mapping        -- img_<N>.png <-> clean file, verified by exact pixel match on the
                       images where no augmentation op fired (maxdiff <= 2; a wrong pair
                       differs by >100), plus a negative control
  4. severity       -- untouched fraction and mean absolute deviation per distribution

Read-only. Never writes into the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".JPG", ".JPEG", ".PNG")
IMG_LOWER = (".jpg", ".jpeg", ".png")

# label -> directory. `enhanced` is the Moderate tier; the historical
# `evaluation-results/medium_augment_evaluation_report.csv` uses the same ordering.
DISTRIBUTIONS = (
    ("clean",    "test"),
    ("easy",     "augmented_test_images_easy"),
    ("moderate", "augmented_test_images_enhanced"),
    ("hard",     "augmented_test_images_hardest"),
)

EXPECTED_N = 8335
EXPECTED_CLASSES = 39
MATCH_TOL = 2          # a true pair differs by <=1 from float round-trip; a wrong pair by >100

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def listing_sha256(root: str, classes: list[str]) -> tuple[str, int, dict]:
    """Same construction as aetfpe.data.dataset_fingerprint, kept dependency-free."""
    h = hashlib.sha256()
    total = 0
    per_class = {}
    for c in classes:
        d = os.path.join(root, c)
        files = sorted(f for f in os.listdir(d) if f.endswith(IMG_EXT)) if os.path.isdir(d) else []
        per_class[c] = len(files)
        total += len(files)
        h.update(c.encode())
        for f in files:
            h.update(f.encode())
    return h.hexdigest(), total, per_class


def noop_render(path: str) -> np.ndarray:
    """What augment_pt.py writes when no stochastic op fires: Resize(224) + normalize
    round-trip + uint8 truncation."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    y = ((img.astype(np.float32) / 255.0 - _MEAN) / _STD) * _STD + _MEAN
    return (np.clip(y, 0, 1) * 255).astype(np.uint8)


def build_mapping(clean_root: str, classes: list[str]) -> dict:
    """Freeze `img_<global>.png` -> clean relative path, per augment_pt.py's ordering."""
    order, start, cum = {}, {}, 0
    for c in classes:
        files = [f for f in os.listdir(os.path.join(clean_root, c)) if f.lower().endswith(IMG_LOWER)]
        order[c] = files                 # os.listdir order, deliberately NOT sorted
        start[c] = cum
        cum += len(files)
    mapping = {}
    for c in classes:
        for j, f in enumerate(order[c]):
            mapping[f"{c}/img_{start[c] + j}.png"] = f"{c}/{f}"
    return {"total": cum, "class_start": start, "listdir_order": order, "map": mapping}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset", "Plant_leaf_diseases_dataset"))
    ap.add_argument("--out", default="results/eval_integrity")
    ap.add_argument("--verify-limit", type=int, default=400,
                    help="images per distribution to pixel-verify the mapping on")
    args = ap.parse_args()

    R = args.data_root
    os.makedirs(args.out, exist_ok=True)
    report: dict = {"data_root": R, "distributions": {}, "checks": {}}
    failures: list[str] = []

    clean_root = os.path.join(R, "test")
    classes = sorted(d for d in os.listdir(clean_root) if os.path.isdir(os.path.join(clean_root, d)))

    # ---- 1. structure ------------------------------------------------------ #
    print("== 1. structure ==")
    ref_classes = ref_per_class = None
    for label, sub in DISTRIBUTIONS:
        root = os.path.join(R, sub)
        if not os.path.isdir(root):
            failures.append(f"{label}: directory missing ({root})")
            continue
        cs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        sha, n, per_class = listing_sha256(root, cs)
        report["distributions"][label] = {
            "dir": sub, "num_images": n, "num_classes": len(cs),
            "listing_sha256": sha, "per_class": per_class,
        }
        if n != EXPECTED_N:
            failures.append(f"{label}: {n} images, expected {EXPECTED_N}")
        if len(cs) != EXPECTED_CLASSES:
            failures.append(f"{label}: {len(cs)} classes, expected {EXPECTED_CLASSES}")
        if ref_classes is None:
            ref_classes, ref_per_class = cs, per_class
        else:
            if cs != ref_classes:
                failures.append(f"{label}: class ordering differs from clean")
            if per_class != ref_per_class:
                failures.append(f"{label}: per-class supports differ from clean")
        print(f"  {label:9s} n={n:5d} classes={len(cs):3d} listing={sha[:16]}…")

    # ---- 2. distinctness --------------------------------------------------- #
    print("\n== 2. distinctness (content hashes) ==")
    hashes: dict[str, dict[str, str]] = {}
    for label, sub in DISTRIBUTIONS[1:]:
        root = os.path.join(R, sub)
        h = {}
        for c in classes:
            d = os.path.join(root, c)
            for f in os.listdir(d):
                if f.endswith(".png"):
                    h[f"{c}/{f}"] = sha256_file(os.path.join(d, f))
        hashes[label] = h
        print(f"  {label:9s} {len(h)} files, {len(set(h.values()))} unique content hashes")
    labels = [l for l, _ in DISTRIBUTIONS[1:]]
    overlaps = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            same = sum(1 for k in hashes[a] if hashes[a][k] == hashes[b][k])
            overlaps[f"{a}_vs_{b}"] = same
            frac = same / max(len(hashes[a]), 1)
            print(f"  {a:9s} vs {b:9s}: {same:5d} byte-identical ({frac:.2%})")
            if frac > 0.10:
                failures.append(f"{a} and {b} share {frac:.1%} identical files -- not distinct tiers")
    report["checks"]["pairwise_identical_files"] = overlaps

    # ---- 3. mapping -------------------------------------------------------- #
    print("\n== 3. clean <-> augmented mapping ==")
    m = build_mapping(clean_root, classes)
    if m["total"] != EXPECTED_N:
        failures.append(f"mapping covers {m['total']} images, expected {EXPECTED_N}")

    keys = sorted(hashes[labels[0]].keys(), key=lambda k: (k.split("/")[0], int(re.search(r"img_(\d+)", k).group(1))))
    step = max(1, len(keys) // args.verify_limit)
    probe = keys[::step][:args.verify_limit]

    confirmed = fired = contradicted = 0
    neg_control_ok = 0
    for k in probe:
        c, fn = k.split("/")
        rel = m["map"].get(k)
        if rel is None:
            failures.append(f"mapping has no entry for {k}")
            continue
        aug = np.array(Image.open(os.path.join(R, "augmented_test_images_easy", c, fn)).convert("RGB"))
        d = np.abs(noop_render(os.path.join(clean_root, rel)).astype(np.int16) - aug.astype(np.int16))
        if d.max() <= MATCH_TOL:
            confirmed += 1
            # negative control: a different image in the same class must NOT match
            others = [v for kk, v in m["map"].items() if kk.startswith(c + "/") and v != rel]
            if others:
                alt = others[len(others) // 2]
                dd = np.abs(noop_render(os.path.join(clean_root, alt)).astype(np.int16) - aug.astype(np.int16))
                if dd.max() > MATCH_TOL:
                    neg_control_ok += 1
                else:
                    contradicted += 1
                    failures.append(f"negative control failed: {k} also matches {alt}")
        else:
            fired += 1

    print(f"  probed {len(probe)} images of `easy`")
    print(f"    exact match at predicted clean file : {confirmed}")
    print(f"    augmentation fired (no exact match) : {fired}")
    print(f"    negative controls passed            : {neg_control_ok}/{confirmed}")
    verdict = ("PROVEN" if confirmed > 0 and contradicted == 0 and neg_control_ok == confirmed
               else "NOT PROVEN")
    print(f"  MAPPING VERDICT: {verdict}")
    report["checks"]["mapping"] = {
        "verdict": verdict, "probed": len(probe), "confirmed_exact": confirmed,
        "augmentation_fired": fired, "negative_controls_passed": neg_control_ok,
        "contradictions": contradicted, "tolerance_maxdiff": MATCH_TOL,
        "rule": "img_<global>.png in class c  <->  os.listdir(test/c)[global - class_start(c)]; "
                "classes sorted; class_start = cumulative counts in sorted-class order",
    }

    # ---- 4. severity characterisation -------------------------------------- #
    print("\n== 4. severity characterisation ==")
    sev = {}
    for label, sub in DISTRIBUTIONS[1:]:
        untouched = 0
        mads = []
        for k in probe:
            c, fn = k.split("/")
            rel = m["map"].get(k)
            if rel is None:
                continue
            ref = noop_render(os.path.join(clean_root, rel)).astype(np.int16)
            aug = np.array(Image.open(os.path.join(R, sub, c, fn)).convert("RGB")).astype(np.int16)
            d = np.abs(ref - aug)
            if d.max() <= MATCH_TOL:
                untouched += 1
            mads.append(float(d.mean()))
        sev[label] = {
            "untouched_fraction": round(untouched / max(len(probe), 1), 4),
            "mean_abs_deviation": round(float(np.mean(mads)), 3),
            "median_abs_deviation": round(float(np.median(mads)), 3),
            "n_probed": len(probe),
        }
        print(f"  {label:9s} untouched={sev[label]['untouched_fraction']:.2%}  "
              f"MAD={sev[label]['mean_abs_deviation']:7.3f}  "
              f"median={sev[label]['median_abs_deviation']:7.3f}")
    report["checks"]["severity"] = sev
    order_ok = (sev["easy"]["mean_abs_deviation"] <= sev["moderate"]["mean_abs_deviation"]
                <= sev["hard"]["mean_abs_deviation"])
    print(f"  monotone easy <= moderate <= hard: {order_ok}")
    report["checks"]["severity_monotone"] = bool(order_ok)
    if not order_ok:
        print("  NOTE: tiers are not monotone in mean absolute deviation -- record, do not relabel.")

    # ---- write ------------------------------------------------------------- #
    with open(os.path.join(args.out, "clean_augmented_mapping.json"), "w") as fh:
        json.dump({"rule": report["checks"]["mapping"]["rule"],
                   "verdict": verdict, "class_start": m["class_start"],
                   "map": m["map"]}, fh, indent=2)
    with open(os.path.join(args.out, "augmented_content_hashes.json"), "w") as fh:
        json.dump(hashes, fh)
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    with open(os.path.join(args.out, "distribution_verification.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nwrote {args.out}/{{distribution_verification,clean_augmented_mapping,augmented_content_hashes}}.json")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  *", f)
        return 1
    print("\nSTATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
