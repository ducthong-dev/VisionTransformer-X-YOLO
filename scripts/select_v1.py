#!/usr/bin/env python
"""Stage V1 selection -- applies the frozen rule mechanically, after the three
pre-registered candidate runs complete (PROVENANCE_MATRIX.md Section 3b).

Does not train anything. Reads each candidate's train_summary.json (validation
top-1) and, if present, its corruptions_val evaluation, and applies the rule
exactly as frozen:

    highest validation top-1 wins;
    if two candidates are within 0.5 percentage points, choose the simpler
    configuration (lower ae_loss_weight, then fewer warm-up epochs).

No other selection logic is permitted. This script will refuse to run against
any candidate set other than the three frozen ones.

    python scripts/select_v1.py \\
        --v1a results/validation/V1_w10_warm3 \\
        --v1b results/validation/V1_w1_warm3 \\
        --v1c results/validation/V1_w10_warm0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import resolve_roots  # noqa: E402

# FROZEN candidates. Any run whose recorded (ae_loss_weight, ae_warmup_epochs)
# does not match one of these three is rejected.
CANDIDATES = {
    "V1a": {"ae_loss_weight": 10, "ae_warmup_epochs": 3},
    "V1b": {"ae_loss_weight": 1, "ae_warmup_epochs": 3},
    "V1c": {"ae_loss_weight": 10, "ae_warmup_epochs": 0},
}
TIE_BAND_PP = 0.5


def load_run(run_dir: str) -> dict:
    summary_path = os.path.join(run_dir, "train_summary.json")
    if not os.path.exists(summary_path):
        raise SystemExit(f"no train_summary.json in {run_dir} -- has this candidate finished training?")
    with open(summary_path) as fh:
        summary = json.load(fh)

    protocol = summary.get("protocol", {})
    weight = protocol.get("ae_loss_weight")
    warmup = protocol.get("ae_warmup_epochs")
    val_top1 = summary.get("best_val_top1")
    if val_top1 is None:
        raise SystemExit(f"{run_dir}: train_summary.json has no best_val_top1")

    # optional: mean corrupted validation top-1, if evaluate.py has been run
    # against corruptions_val for this candidate
    corr = None
    corr_csv = os.path.join(run_dir, "test_corruptions.csv")
    if os.path.exists(corr_csv):
        import csv as _csv

        with open(corr_csv) as fh:
            vals = [float(r["top1"]) for r in _csv.DictReader(fh)
                   if r["corruption"] not in ("clean", "clean_testsplit")]
        if vals:
            corr = sum(vals) / len(vals)

    return {
        "run_dir": run_dir,
        "ae_loss_weight": weight,
        "ae_warmup_epochs": warmup,
        "val_top1": val_top1,
        "mean_corrupted_val_top1": corr,
        "train_seconds": summary.get("train_seconds"),
    }


def identify(run: dict) -> str:
    for label, spec in CANDIDATES.items():
        if run["ae_loss_weight"] == spec["ae_loss_weight"] and run["ae_warmup_epochs"] == spec["ae_warmup_epochs"]:
            return label
    raise SystemExit(
        f"{run['run_dir']}: (ae_loss_weight={run['ae_loss_weight']}, "
        f"ae_warmup_epochs={run['ae_warmup_epochs']}) does not match any of the "
        f"three frozen V1 candidates {CANDIDATES}. Stage V1 permits no fourth "
        f"candidate (PROVENANCE_MATRIX.md Section 3b)."
    )


def simplicity_rank(label: str) -> int:
    """Lower is simpler. Rule: lower weight first, then fewer warm-up epochs."""
    spec = CANDIDATES[label]
    return (spec["ae_loss_weight"], spec["ae_warmup_epochs"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1a", required=True, help="run dir for ae_loss_weight=10, warmup=3")
    ap.add_argument("--v1b", required=True, help="run dir for ae_loss_weight=1, warmup=3")
    ap.add_argument("--v1c", required=True, help="run dir for ae_loss_weight=10, warmup=0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    args.out = args.out or os.path.join(resolve_roots()["OUTPUT_ROOT"], "v1", "selection.json")

    runs = {}
    for arg_val in (args.v1a, args.v1b, args.v1c):
        r = load_run(arg_val)
        label = identify(r)
        if label in runs:
            raise SystemExit(f"duplicate candidate {label}: {runs[label]['run_dir']} and {arg_val}")
        runs[label] = r

    missing = set(CANDIDATES) - set(runs)
    if missing:
        raise SystemExit(f"missing candidate(s): {sorted(missing)}. All three must be provided.")

    print("candidate  weight  warmup  val_top1  mean_corrupted_val_top1  train_s")
    for label in ("V1a", "V1b", "V1c"):
        r = runs[label]
        c = f"{r['mean_corrupted_val_top1']:.4f}" if r["mean_corrupted_val_top1"] is not None else "n/a"
        print(f"{label:<10} {r['ae_loss_weight']:<7} {r['ae_warmup_epochs']:<7} "
              f"{r['val_top1']:.4f}    {c:<24} {r.get('train_seconds', '?')}")

    best_label = max(runs, key=lambda k: runs[k]["val_top1"])
    best_top1 = runs[best_label]["val_top1"]

    within_band = [
        label for label in runs
        if (best_top1 - runs[label]["val_top1"]) * 100 <= TIE_BAND_PP
    ]

    if len(within_band) > 1:
        selected = min(within_band, key=simplicity_rank)
        rule_applied = (f"tie within {TIE_BAND_PP}pp among {within_band}; "
                        f"selected simplest: {selected}")
    else:
        selected = best_label
        rule_applied = f"clear winner by validation top-1: {selected}"

    result = {
        "candidates": runs,
        "selection_rule": "highest validation top-1; ties within 0.5pp -> simplest config",
        "rule_applied": rule_applied,
        "selected": selected,
        "selected_ae_loss_weight": CANDIDATES[selected]["ae_loss_weight"],
        "selected_ae_warmup_epochs": CANDIDATES[selected]["ae_warmup_epochs"],
        "frozen_after_this": True,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n{rule_applied}")
    print(f"SELECTED: {selected}  "
          f"(ae_loss_weight={CANDIDATES[selected]['ae_loss_weight']}, "
          f"ae_warmup_epochs={CANDIDATES[selected]['ae_warmup_epochs']})")
    print(f"\nwrote {args.out}")
    print("\nNo further tuning is permitted after this point unless a training run "
          "is demonstrably broken (diverges, NaNs, or chance-level accuracy) -- "
          "see SCIENTIFIC_PROTOCOL_FROZEN.md.")
    print(f"\nNext: set ae_loss_weight={CANDIDATES[selected]['ae_loss_weight']} and "
          f"ae_warmup_epochs={CANDIDATES[selected]['ae_warmup_epochs']} in configs/_base.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
