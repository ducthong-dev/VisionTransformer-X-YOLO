#!/usr/bin/env python
"""Evaluate verified checkpoints on the four frozen evaluation distributions.

    python scripts/evaluate_distributions.py                       # priority order, resumable
    python scripts/evaluate_distributions.py --runs A0 E5 --device mps

`scripts/evaluate.py` is left untouched: it targets the generated corruption tree, and
its historical outputs must stay reproducible. This script targets the four pre-made
distributions that ship with the dataset (see docs/CORRUPTION_PROTOCOL.md).

Primary output is PREDICTION-LEVEL, one gzipped CSV per (physical run, distribution):

    run_id, distribution, sample_id, relative_path, ground_truth_index,
    ground_truth_class, predicted_index, predicted_class, confidence,
    top5_indices, top5_scores, correct_top1, correct_top5

`sample_id` is the CLEAN SOURCE image's relative path in every distribution, resolved
through the proven mapping in results/eval_integrity/clean_augmented_mapping.json. That
is what makes cross-distribution paired analysis possible: the same sample_id identifies
the same underlying photograph in clean, easy, moderate and hard.

Aggregates are DERIVED from these records, never computed alongside them.

Inference only: torch.no_grad(), no checkpoint written, no test image influences model
selection (the checkpoint is the frozen best_val_top1 artifact). Fully resumable --
a completed (run, distribution) pair is skipped on restart.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import environment_info, pick_device  # noqa: E402
from aetfpe.data import LeafDataset, list_classes  # noqa: E402
from aetfpe.metrics import summarize  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.seeding import seed_everything  # noqa: E402

# label -> dataset subdirectory. `enhanced` IS the Moderate tier; the mapping is stated
# in every output rather than applied silently. See docs/CORRUPTION_PROTOCOL.md.
DISTRIBUTIONS = (
    ("clean",    "test"),
    ("easy",     "augmented_test_images_easy"),
    ("moderate", "augmented_test_images_enhanced"),
    ("hard",     "augmented_test_images_hardest"),
)
LABEL_MAPPING_NOTE = {
    "clean": "test (original JPEG images)",
    "easy": "augmented_test_images_easy",
    "moderate": "augmented_test_images_enhanced  <-- 'enhanced' is the Moderate tier",
    "hard": "augmented_test_images_hardest",
}

# Explicit evaluation waves. Wave 1 is the decision-critical evidence checkpoint and is
# never delayed by later arrivals. Wave 2 completes the component-ablation sequence
# A0 -> A1 -> A2 -> A3 -> A4 -> A5 (A0 and A5 are already covered by wave 1).
WAVE_1_DECISION_CRITICAL = ["A0", "A5", "D1", "E5", "B2"]
WAVE_2_COMPONENT_ABLATION = ["A1", "A2", "A3", "A4"]
WAVE_3_REMAINING = ["F2", "F4", "F1", "E3", "E7", "M1", "M2", "M3", "B1", "B3"]
PRIORITY = WAVE_1_DECISION_CRITICAL
SECONDARY = WAVE_2_COMPONENT_ABLATION + WAVE_3_REMAINING

ALIASES = {"F3": "A3", "F5": "A5", "F5_clean": "D1"}

FIELDS = ["run_id", "distribution", "sample_id", "relative_path",
          "ground_truth_index", "ground_truth_class", "predicted_index",
          "predicted_class", "confidence", "top5_indices", "top5_scores",
          "correct_top1", "correct_top5"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_mapping(path: str) -> dict:
    """augmented 'class/img_N.png' -> clean 'class/file.JPG'. Frozen, not recomputed."""
    if not os.path.exists(path):
        raise SystemExit(
            f"ERROR: {path} not found.\n"
            "Run scripts/verify_eval_distributions.py first -- cross-distribution pairing\n"
            "depends on the proven mapping being frozen to disk.")
    d = json.load(open(path))
    if d.get("verdict") != "PROVEN":
        raise SystemExit(f"ERROR: mapping verdict is {d.get('verdict')!r}, refusing to use it.")
    return d["map"]


@torch.no_grad()
def run_one(model, root, classes, device, img_size, batch_size, workers,
            run_id, label, mapping) -> tuple[list[dict], dict]:
    ds = LeafDataset(root, classes, img_size, train=False, return_path=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    rows: list[dict] = []
    all_logits, all_y = [], []
    for x, y, paths in dl:
        logits = model(x.to(device)).float().cpu()
        all_logits.append(logits.numpy())
        all_y.append(y.numpy())
        probs = torch.softmax(logits, dim=1)
        top5s, top5i = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
        for j, p in enumerate(paths):
            rel = os.path.relpath(p, root)
            gt = int(y[j])
            pi = int(top5i[j, 0])
            t5 = [int(v) for v in top5i[j]]
            rows.append({
                "run_id": run_id,
                "distribution": label,
                # the clean source image identifies the sample in EVERY distribution
                "sample_id": mapping.get(rel, rel) if label != "clean" else rel,
                "relative_path": rel,
                "ground_truth_index": gt,
                "ground_truth_class": classes[gt],
                "predicted_index": pi,
                "predicted_class": classes[pi],
                "confidence": round(float(top5s[j, 0]), 6),
                "top5_indices": "|".join(str(v) for v in t5),
                "top5_scores": "|".join(f"{float(v):.6f}" for v in top5s[j]),
                "correct_top1": int(pi == gt),
                "correct_top5": int(gt in t5),
            })
    agg = summarize(np.concatenate(all_logits), np.concatenate(all_y), classes)
    return rows, agg


def write_rows(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)          # atomic: a killed run never leaves a half file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", default="results/campaign/scientific")
    ap.add_argument("--verification", default="results/campaign/checkpoint_verification.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="results/evaluation")
    ap.add_argument("--mapping", default="docs/evidence/clean_augmented_mapping.json",
                    help="FROZEN mapping. Never regenerate it on another machine: os.listdir\n                          order is filesystem-dependent and a rebuild would be wrong.")
    ap.add_argument("--runs", nargs="*", default=None, help="explicit run ids; default = priority order")
    ap.add_argument("--distributions", nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="recompute completed pairs")
    args = ap.parse_args()

    if args.data_root is None:
        import yaml
        loc = yaml.safe_load(open("configs/local.yaml"))
        args.data_root = loc["env"]["DATA_ROOT"]

    if not os.path.exists(args.verification):
        raise SystemExit(
            f"ERROR: {args.verification} not found.\n"
            "Checkpoints must be verified before evaluation. Run scripts/verify_checkpoints.py.")
    ver = json.load(open(args.verification))
    accepted = set(ver.get("accepted") or [])
    by_id = {r["run_id"]: r for r in ver.get("runs", [])}
    if not accepted:
        raise SystemExit("ERROR: no ACCEPTED checkpoints in the verification report. Nothing to evaluate.")

    order = args.runs or [r for r in PRIORITY + SECONDARY if r in accepted]
    for r in order:
        if r in ALIASES:
            raise SystemExit(f"ERROR: {r} is an alias of {ALIASES[r]}; evaluate the physical run instead.")
        if r not in accepted:
            raise SystemExit(f"ERROR: {r} is not an ACCEPTED checkpoint. Refusing to evaluate it.")

    dists = [(l, s) for l, s in DISTRIBUTIONS if args.distributions is None or l in args.distributions]
    mapping = load_mapping(args.mapping)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    clean_classes = list_classes(os.path.join(args.data_root, "test"))
    print(f"device={device}  data_root={args.data_root}")
    print(f"runs={order}")
    print(f"distributions={[l for l, _ in dists]}\n")

    for rid in order:
        run_out = os.path.join(args.out, rid)
        os.makedirs(run_out, exist_ok=True)
        pending = [(l, s) for l, s in dists
                   if args.force or not os.path.exists(os.path.join(run_out, f"predictions_{l}.csv.gz"))]
        if not pending:
            print(f"[{rid}] all distributions already done -- skipping (resume)")
            continue

        ck_path = os.path.join(args.campaign_root, rid, "checkpoint.pt")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg, classes = ck["cfg"], list(ck["classes"])
        if classes != clean_classes:
            raise SystemExit(f"ERROR: {rid} class ordering differs from the evaluation dataset.")
        img_size = int((cfg.get("protocol") or {}).get("img_size", 224))
        seed_everything(int((cfg.get("protocol") or {}).get("seed", 0)))
        model = build_model(cfg["model"])
        model.load_state_dict(ck["model"], strict=True)
        model = model.to(device).eval()
        print(f"[{rid}] epoch {ck['epoch']} val_top1={float(ck['val_top1']):.7f}  "
              f"pending={[l for l, _ in pending]}")

        summary_path = os.path.join(run_out, "eval_distributions.json")
        meta = json.load(open(summary_path)) if os.path.exists(summary_path) else {
            "run_id": rid,
            "checkpoint": ck_path,
            "checkpoint_sha256": by_id.get(rid, {}).get("checkpoint_sha256") or sha256_file(ck_path),
            "selected_epoch": int(ck["epoch"]),
            "selected_val_top1": float(ck["val_top1"]),
            "label_mapping": LABEL_MAPPING_NOTE,
            "mapping_file": args.mapping,
            "data_root": args.data_root,
            "environment": environment_info(),
            "device": device,
            "batch_size": args.batch_size,
            "img_size": img_size,
            "note": ("Aggregates below are derived from the prediction-level records. "
                     "Training used a single seed (0); no multi-seed claim is supported."),
            "distributions": {},
        }

        for label, sub in pending:
            root = os.path.join(args.data_root, sub)
            t0 = time.time()
            rows, agg = run_one(model, root, classes, device, img_size,
                                args.batch_size, args.num_workers, rid, label, mapping)
            pred_path = os.path.join(run_out, f"predictions_{label}.csv.gz")
            write_rows(pred_path, rows)
            with open(os.path.join(run_out, f"aggregate_{label}.json"), "w") as fh:
                json.dump(agg, fh, indent=2)
            meta["distributions"][label] = {
                "dir": sub, "num_images": agg["overall"]["num_images"],
                "top1": agg["overall"]["top1"], "top5": agg["overall"]["top5"],
                "predictions": os.path.basename(pred_path),
                "predictions_sha256": sha256_file(pred_path),
                "seconds": round(time.time() - t0, 1),
            }
            with open(summary_path, "w") as fh:
                json.dump(meta, fh, indent=2, default=str)
            print(f"    {label:9s} top1={agg['overall']['top1']:.4f} "
                  f"top5={agg['overall']['top5']:.4f} n={agg['overall']['num_images']} "
                  f"({meta['distributions'][label]['seconds']}s)")
        del model

    print(f"\nwrote prediction-level records under {args.out}/<run>/predictions_<dist>.csv.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
