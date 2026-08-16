#!/usr/bin/env python
"""Calibration-only evaluation: validation split + validation corruptions ONLY.

Exists because `scripts/evaluate.py` unconditionally evaluates the clean TEST
split (it was written for Stage 8, where that is exactly what's wanted) and has
no flag to skip it. Using `evaluate.py` during Stage V1 -- even just to compute
`mean_corrupted_val_top1` -- would have touched the frozen test split's images
during a model-selection stage, which SCIENTIFIC_PROTOCOL_FROZEN.md prohibits
outright. That defect is fixed here structurally, not with a flag that could be
forgotten: this file's code never constructs a test-split path at all -- it has
no reference to `cfg["data"]["test_split"]` or `cfg["data"]["root"]`, and reads
only PNG trees that already exist under `--corruption-root`.

As a second, independent guard, it refuses to run unless the corruption root's
own `generation_environment.json` bundle records `"split": "val"` -- tying the
check to the generator's own metadata rather than a directory-naming
convention, so renaming a directory can't fool it.

    python scripts/evaluate_calibration.py \\
        --run results/validation/V1a_w10_warm3 \\
        --corruption-root results/corruptions_val

Writes, into the run directory (filenames deliberately distinct from
evaluate.py's test_clean.json/test_corruptions.csv/eval_summary.json, so
calibration output can never be mistaken for or overwritten by the Stage 8
final evaluation):

    val_clean.json / val_clean.csv       -- validation split, uncorrupted
    val_corruptions.csv                  -- one row per corruption x severity
    calibration_eval_summary.json        -- checksums, environment, all rows
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import environment_info, pick_device, resolve_roots  # noqa: E402
from aetfpe.data import LeafDataset  # noqa: E402
from aetfpe.metrics import summarize  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.seeding import seed_everything, sha256_file  # noqa: E402


@torch.no_grad()
def eval_dir(model, root, classes, device, img_size, batch_size, workers):
    ds = LeafDataset(root, classes, img_size, train=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    logits, ys = [], []
    for x, y in dl:
        logits.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    return summarize(np.concatenate(logits), np.concatenate(ys), classes)


def discover(corruption_root: str) -> list[tuple[str, str, str]]:
    out = []
    for entry in sorted(os.listdir(corruption_root)):
        p = os.path.join(corruption_root, entry)
        if not os.path.isdir(p):
            continue
        if entry == "clean":
            out.append(("clean", "none", p))
            continue
        for sev in sorted(os.listdir(p)):
            sp = os.path.join(p, sev)
            if os.path.isdir(sp):
                out.append((entry, sev, sp))
    return out


def assert_is_validation_root(corruption_root: str) -> dict:
    bundle_path = os.path.join(corruption_root, "generation_environment.json")
    if not os.path.exists(bundle_path):
        raise SystemExit(
            f"no generation_environment.json in {corruption_root!r} -- cannot confirm "
            f"this is a validation corruption set. Refusing to run. "
            f"Run scripts/generate_corruptions.py --split val first."
        )
    with open(bundle_path) as fh:
        bundle = json.load(fh)
    if bundle.get("split") != "val":
        raise SystemExit(
            f"REFUSING TO RUN: {corruption_root!r} was generated with "
            f"split={bundle.get('split')!r}, not 'val'. Stage V1 calibration must "
            f"never touch the frozen test benchmark. If you meant to point at "
            f"corruptions_val, check --corruption-root."
        )
    return bundle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corruption-root", default=None,
                    help="default: ${OUTPUT_ROOT}/corruptions_val")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    args.corruption_root = args.corruption_root or os.path.join(
        resolve_roots()["OUTPUT_ROOT"], "corruptions_val")

    bundle = assert_is_validation_root(args.corruption_root)

    ckpt_path = os.path.join(args.run, "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg, classes = ckpt["cfg"], ckpt["classes"]
    protocol_size = int((cfg.get("protocol") or {}).get("img_size", 224))

    seed_everything(int((cfg.get("protocol") or {}).get("seed", 0)))
    device = pick_device(args.device)

    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    name = cfg.get("name", os.path.basename(args.run))
    print(f"[{name}] loaded epoch {ckpt.get('epoch')} val_top1={ckpt.get('val_top1'):.4f} device={device}")
    print(f"corruption root: {args.corruption_root} (confirmed split=val)")

    rows = []
    clean_result = None
    for corruption, severity, path in discover(args.corruption_root):
        res = eval_dir(model, path, classes, device, protocol_size, args.batch_size, args.num_workers)
        if corruption == "clean":
            clean_result = res
        rows.append({"corruption": corruption, "severity": severity, **res["overall"]})
        print(f"  {corruption}/{severity}: top1={res['overall']['top1']:.4f} "
              f"top5={res['overall']['top5']:.4f} n={res['overall']['num_images']}")

    if clean_result is None:
        raise SystemExit(f"no 'clean' directory found under {args.corruption_root}")

    with open(os.path.join(args.run, "val_clean.json"), "w") as fh:
        json.dump(clean_result, fh, indent=2)
    with open(os.path.join(args.run, "val_clean.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(clean_result["per_class"][0].keys()))
        w.writeheader()
        w.writerows(clean_result["per_class"])

    with open(os.path.join(args.run, "val_corruptions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    manifest = os.path.join(args.corruption_root, "corruption_manifest.csv")
    with open(os.path.join(args.run, "calibration_eval_summary.json"), "w") as fh:
        json.dump({
            "name": name,
            "purpose": "Stage V1 / model-selection calibration -- NOT the frozen test benchmark",
            "checkpoint": ckpt_path,
            "checkpoint_sha256": hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest(),
            "corruption_root": args.corruption_root,
            "corruption_root_split": bundle.get("split"),
            "corruption_manifest_sha256": sha256_file(manifest) if os.path.exists(manifest) else None,
            "environment": environment_info(),
            "device": device,
            "rows": rows,
        }, fh, indent=2, default=str)

    print(f"[{name}] wrote {len(rows)} calibration rows -> {args.run} "
          f"(val_clean.*, val_corruptions.csv, calibration_eval_summary.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
