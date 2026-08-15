#!/usr/bin/env python
"""G3 -- baseline reproduction gate (COLAB_CAMPAIGN_PLAN.md / SCIENTIFIC_PROTOCOL_FROZEN.md).

Applies the frozen threshold mechanically to A0_baseline_rgb's completed
training run. Does not train anything -- run this immediately after:

    python scripts/train.py --config configs/baseline_rgb.yaml

    python scripts/check_g3_gate.py \\
        --run "${OUTPUT_ROOT}/ablation/A0_baseline_rgb"

A0's own run directory (`${OUTPUT_ROOT}/ablation/A0_baseline_rgb/`) IS the G3
artifact set: it already contains config.yaml, environment.json, metrics.csv
(the training log), and checkpoint.pt, so this script does not duplicate a
~3 MB checkpoint into a second location. It writes a small derived verdict to
results/g3/gate_result.json that references the run directory and copies the
lightweight provenance files (config, environment, metrics) alongside it, plus
the dataset manifest hash, satisfying "Save: results/g3/ including config,
checkpoint[-reference], metrics, training log, environment, dataset manifest
hash" without a second unique training.

Thresholds are hardcoded, not parameters -- SCIENTIFIC_PROTOCOL_FROZEN.md
Section "Frozen numeric thresholds" says explicitly: "Not revisable after
observing Colab output."
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import environment_info, resolve_roots  # noqa: E402

# FROZEN. Do not edit after seeing a result.
HISTORICAL_REFERENCE_TOP1 = 0.9969
HISTORICAL_REFERENCE_TOP5 = 0.9996
PASS_THRESHOLD = 0.990
CONDITIONAL_THRESHOLD = 0.980
MIN_TOP5 = 0.998


def verdict(top1: float, top5: float | None) -> tuple[str, str]:
    if top1 < CONDITIONAL_THRESHOLD:
        return "FAIL", (
            f"val top-1 {top1:.4f} < {CONDITIONAL_THRESHOLD} -- STOP THE CAMPAIGN. "
            f"The protocol reconstruction does not reproduce the historical baseline "
            f"({HISTORICAL_REFERENCE_TOP1}). Do not proceed to any other stage."
        )
    if top1 < PASS_THRESHOLD:
        return "CONDITIONAL", (
            f"val top-1 {top1:.4f} in [{CONDITIONAL_THRESHOLD}, {PASS_THRESHOLD}) -- "
            f"STOP FOR REVIEW. May proceed only if the deviation from "
            f"{HISTORICAL_REFERENCE_TOP1} is investigated and disclosed in the "
            f"manuscript's protocol section."
        )
    if top5 is not None and top5 < MIN_TOP5:
        return "CONDITIONAL", (
            f"val top-1 {top1:.4f} passes, but top-5 {top5:.4f} < {MIN_TOP5}. "
            f"Investigate before proceeding."
        )
    return "PASS", f"val top-1 {top1:.4f} >= {PASS_THRESHOLD}. Proceed to Stage 4."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help="A0's run dir; default ${OUTPUT_ROOT}/ablation/A0_baseline_rgb")
    ap.add_argument("--dataset-summary", default=None,
                    help="path to dataset_summary.json from Stage 1, for the manifest hash")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    roots = resolve_roots()
    run_dir = args.run or os.path.join(roots["OUTPUT_ROOT"], "ablation", "A0_baseline_rgb")
    out_dir = args.out_dir or os.path.join(roots["OUTPUT_ROOT"], "g3")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(run_dir, "train_summary.json")
    if not os.path.exists(summary_path):
        raise SystemExit(f"no train_summary.json in {run_dir} -- has A0 finished training?")
    with open(summary_path) as fh:
        summary = json.load(fh)

    top1 = summary.get("best_val_top1")
    if top1 is None:
        raise SystemExit(f"{summary_path} has no best_val_top1")

    # top-5 is not currently persisted in train_summary.json; pull it from the
    # last epoch of metrics.csv if present.
    top5 = None
    metrics_path = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        import csv as _csv

        with open(metrics_path) as fh:
            rows = list(_csv.DictReader(fh))
        if rows and "val_top5" in rows[0]:
            best_row = max(rows, key=lambda r: float(r["val_top5"]))
            top5 = float(best_row["val_top5"])

    status, message = verdict(top1, top5)

    for fname in ("config.yaml", "environment.json", "metrics.csv"):
        src = os.path.join(run_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))

    dataset_manifest_sha256 = None
    if args.dataset_summary and os.path.exists(args.dataset_summary):
        with open(args.dataset_summary) as fh:
            dataset_manifest_sha256 = json.load(fh).get("manifest_sha256")

    result = {
        "gate": "G3",
        "run_dir": run_dir,
        "checkpoint": os.path.join(run_dir, "checkpoint.pt"),
        "val_top1": top1,
        "val_top5": top5,
        "historical_reference_top1": HISTORICAL_REFERENCE_TOP1,
        "historical_reference_top5": HISTORICAL_REFERENCE_TOP5,
        "deviation_pp": round((HISTORICAL_REFERENCE_TOP1 - top1) * 100, 4),
        "thresholds": {
            "pass": PASS_THRESHOLD, "conditional": CONDITIONAL_THRESHOLD, "min_top5": MIN_TOP5,
        },
        "status": status,
        "message": message,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "environment": environment_info(),
    }

    with open(os.path.join(out_dir, "gate_result.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    print(f"G3 baseline reproduction: val_top1={top1:.4f}  "
          f"(historical reference {HISTORICAL_REFERENCE_TOP1})")
    print(f"deviation: {result['deviation_pp']:+.3f} pp")
    print(f"\n{status}: {message}")
    print(f"\nwrote {out_dir}/gate_result.json")

    if status == "FAIL":
        return 2
    if status == "CONDITIONAL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
