#!/usr/bin/env python
"""Evaluate a trained arm on the clean test split and on every frozen corruption.

    python scripts/evaluate.py --run results/ablation/A0_baseline_rgb
    python scripts/evaluate.py --run results/... --corruption-root data/corruptions

Reads the corruption manifest so the evaluation is provably against the same
files every other model saw, and records the manifest checksum in the output.
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
from aetfpe.data import LeafDataset, list_classes  # noqa: E402
from aetfpe.metrics import summarize  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.seeding import seed_everything, sha256_file  # noqa: E402


@torch.no_grad()
def eval_dir(model, root, classes, device, img_size, batch_size, workers, limit_per_class=None):
    ds = LeafDataset(root, classes, img_size, train=False)
    if limit_per_class:
        seen: dict[int, int] = {}
        keep = []
        for s in ds.samples:
            if seen.get(s[1], 0) < limit_per_class:
                keep.append(s)
                seen[s[1]] = seen.get(s[1], 0) + 1
        ds.samples = keep
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    logits, ys = [], []
    for x, y in dl:
        logits.append(model(x.to(device)).float().cpu().numpy())
        ys.append(y.numpy())
    return summarize(np.concatenate(logits), np.concatenate(ys), classes)


def discover(corruption_root: str) -> list[tuple[str, str, str]]:
    """-> [(corruption, severity, path)] from the on-disk layout."""
    out = []
    if not os.path.isdir(corruption_root):
        return out
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<group>/<name> directory")
    ap.add_argument("--corruption-root", default=None, help="default: ${OUTPUT_ROOT}/corruptions")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--skip-corruptions", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="smoke-test mode: cap images per class on the clean test split")
    args = ap.parse_args()

    args.corruption_root = args.corruption_root or os.path.join(
        resolve_roots()["OUTPUT_ROOT"], "corruptions")
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

    d = cfg["data"]
    test_root = os.path.join(d["root"], d["test_split"])

    # ---- clean test split ------------------------------------------------- #
    clean = eval_dir(model, test_root, classes, device, protocol_size,
                     args.batch_size, args.num_workers, args.limit_per_class)
    print(f"  clean test: top1={clean['overall']['top1']:.4f} top5={clean['overall']['top5']:.4f} "
          f"n={clean['overall']['num_images']}")

    with open(os.path.join(args.run, "test_clean.json"), "w") as fh:
        json.dump(clean, fh, indent=2)
    with open(os.path.join(args.run, "test_clean.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(clean["per_class"][0].keys()))
        w.writeheader()
        w.writerows(clean["per_class"])

    # ---- frozen corruptions ----------------------------------------------- #
    rows = [{"corruption": "clean_testsplit", "severity": "none",
             **{k: v for k, v in clean["overall"].items()}}]

    manifest = os.path.join(args.corruption_root, "corruption_manifest.csv")
    manifest_hash = sha256_file(manifest) if os.path.exists(manifest) else None

    if not args.skip_corruptions:
        for corruption, severity, path in discover(args.corruption_root):
            res = eval_dir(model, path, classes, device, protocol_size,
                           args.batch_size, args.num_workers)
            rows.append({"corruption": corruption, "severity": severity, **res["overall"]})
            print(f"  {corruption}/{severity}: top1={res['overall']['top1']:.4f} "
                  f"top5={res['overall']['top5']:.4f} n={res['overall']['num_images']}")
            sub = os.path.join(args.run, "per_class", f"{corruption}_{severity}.json")
            os.makedirs(os.path.dirname(sub), exist_ok=True)
            with open(sub, "w") as fh:
                json.dump(res, fh, indent=2)

    with open(os.path.join(args.run, "test_corruptions.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(os.path.join(args.run, "eval_summary.json"), "w") as fh:
        json.dump({
            "name": name,
            "checkpoint": ckpt_path,
            "checkpoint_sha256": hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest(),
            "corruption_root": args.corruption_root,
            "corruption_manifest_sha256": manifest_hash,
            "environment": environment_info(),
            "device": device,
            "rows": rows,
        }, fh, indent=2, default=str)

    print(f"[{name}] wrote {len(rows)} evaluation rows -> {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
