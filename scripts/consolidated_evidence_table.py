#!/usr/bin/env python
"""One consolidated table: 22 logical IDs x 4 evaluation distributions.

    python scripts/consolidated_evidence_table.py

Validation and test numbers NEVER share an ambiguous "Accuracy" column. The columns are
labelled exactly:

    Best Val Top-1 | Clean Test Top-1 | Easy Test Top-1 | Moderate Test Top-1 | Hard Test Top-1

Aggregates are recomputed from the prediction-level records and cross-checked against the
stored aggregate JSON, so the table cannot silently drift from the raw predictions.

Logical aliases (F3->A3, F5->A5, F5_clean->D1) are shown as rows, but every number they
display is attributed to its physical run. Historical Tesla-T4 measurements are never
mixed into these columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import pandas as pd

DISTS = ["clean", "easy", "moderate", "hard"]
COL = {"clean": "Clean Test Top-1", "easy": "Easy Test Top-1",
       "moderate": "Moderate Test Top-1", "hard": "Hard Test Top-1"}
ALIASES = {"F3": "A3", "F5": "A5", "F5_clean": "D1"}

# Logical ID order and the training facts, transcribed from docs/RUN_INVENTORY.md.
INVENTORY = [
    ("A0", "YOLOv8n-cls RGB baseline", 1488247, 1488247, "COMPLETED", 2282.7, 0.9976036424634556),
    ("A1", "PE-only", 1488247, 1488247, "COMPLETED", 2260.4, 0.9968847352024922),
    ("A2", "TF-only (ViT-B/16)", 87289243, 1490587, "INCOMPLETE (26/50)", None, None),
    ("A3", "PE+TF, no AE (= F3)", 87289243, 1490587, "COMPLETED", 6423.0, 0.9966450994488377),
    ("A4", "RGB + image-space AE", 1747290, 1747290, "COMPLETED", 2290.2, 0.9962856458183561),
    ("A5", "Original AE-TFPE full (= F5)", 87549123, 1750467, "COMPLETED", 6270.1, 0.9950874670500839),
    ("B1", "ResNet-50 baseline", 23587943, 23587943, "NOT YET RUN", None, None),
    ("B2", "EfficientNet-B0 baseline", 4057507, 4057507, "COMPLETED", 2634.4, 0.9988018212317278),
    ("B3", "ViT-B/16 baseline", 85828647, 85828647, "NOT YET RUN", None, None),
    ("D1", "Original AE fusion, clean objective", 87549123, 1750467, "COMPLETED", 6741.4, 0.9944883776659478),
    ("E3", "Efficient PE+TF, no AE", 1637947, 1488427, "COMPLETED", 2228.4, 0.9964054636951833),
    ("E5", "Efficient AE-TFPE (C2-28)", 1716586, 1567066, "COMPLETED", 2305.9, 0.9877785765636233),
    ("E7", "Efficient AE-TFPE (C2-7)", 2544634, 1593610, "COMPLETED", 2333.9, 0.8436376707404745),
    ("F1", "Addition fusion", 87289216, 1490560, "COMPLETED", 6178.4, 0.9966450994488377),
    ("F2", "Concatenation fusion", 87289648, 1490992, "COMPLETED", 6171.0, 0.9971243709561467),
    ("F3", "reuses A3", None, None, "LOGICAL_REUSE", None, None),
    ("F4", "Attention fusion", 87289288, 1490632, "COMPLETED", 6465.7, 0.9955667385573927),
    ("F5", "reuses A5", None, None, "LOGICAL_REUSE", None, None),
    ("F5_clean", "reuses D1", None, None, "LOGICAL_REUSE", None, None),
    ("M1", "Legacy LUT control", 1488247, 1488247, "COMPLETED", 2300.4, 0.9958063743110472),
    ("M2", "Photometric (gamma) control", 1488247, 1488247, "COMPLETED", 2284.3, 0.9976036424634556),
    ("M3", "Augmentation control", 1488247, 1488247, "COMPLETED", 2269.2, 0.9968847352024922),
]

NOTES = {
    "E3": "confounded AE control: also changes the fusion space (grid -> image)",
    "F2": "the only arm that modifies the classifier stem",
    "A2": "stalled at epoch 26/50; excluded from evaluation",
    "B1": "skipped by MAX_TRAIN_PARAMS=20,000,000; not scientifically impossible",
    "B3": "skipped by MAX_TRAIN_PARAMS=20,000,000; the backbone the Original method uses",
}


def metrics_from_predictions(eval_root: str, rid: str, dist: str) -> dict | None:
    p = os.path.join(eval_root, rid, f"predictions_{dist}.csv.gz")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, usecols=["ground_truth_index", "predicted_index",
                                 "correct_top1", "correct_top5"])
    gt = df["ground_truth_index"].to_numpy()
    pr = df["predicted_index"].to_numpy()
    k = int(max(gt.max(), pr.max())) + 1
    f1 = []
    for c in range(k):
        tp = int(((pr == c) & (gt == c)).sum())
        fp = int(((pr == c) & (gt != c)).sum())
        fn = int(((pr != c) & (gt == c)).sum())
        d = 2 * tp + fp + fn
        f1.append(0.0 if d == 0 else 2 * tp / d)
    return {"top1": float(df["correct_top1"].mean()),
            "top5": float(df["correct_top5"].mean()),
            "macro_f1": float(sum(f1) / len(f1)),
            "n": int(len(df))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="results/evaluation")
    ap.add_argument("--out-csv", default="results/evaluation/consolidated_evidence_table.csv")
    ap.add_argument("--out-md", default="docs/CONSOLIDATED_EVIDENCE_TABLE.md")
    args = ap.parse_args()

    rows, mismatches = [], []
    for rid, model, params, trainable, status, runtime, bestval in INVENTORY:
        physical = ALIASES.get(rid, rid)
        row = {
            "Logical ID": rid, "Physical run": physical, "Model": model,
            "Total params": params, "Trainable params": trainable,
            "Training status": status, "Training runtime (s)": runtime,
            "Best Val Top-1": bestval,
        }
        for d in DISTS:
            m = metrics_from_predictions(args.eval_root, physical, d)
            row[COL[d]] = None if m is None else round(m["top1"], 6)
            row[f"{COL[d]} (top5)"] = None if m is None else round(m["top5"], 6)
            row[f"{COL[d]} (macroF1)"] = None if m is None else round(m["macro_f1"], 6)
            # cross-check against the stored aggregate
            agg_p = os.path.join(args.eval_root, physical, f"aggregate_{d}.json")
            if m is not None and os.path.exists(agg_p):
                stored = json.load(open(agg_p))["overall"]["top1"]
                if abs(stored - m["top1"]) > 1e-9:
                    mismatches.append(f"{physical}/{d}: stored {stored} vs recomputed {m['top1']}")
        c, h = row.get(COL["clean"]), row.get(COL["hard"])
        row["Robustness drop (clean -> hard)"] = None if (c is None or h is None) else round(c - h, 6)
        row["Notes"] = NOTES.get(rid, "")
        rows.append(row)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(v, nd=4):
        return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else f"{v:,}")

    lines = [
        "# Consolidated Evidence Table", "",
        "Validation and test metrics are in separately labelled columns and are never",
        "combined. Single training seed (0) per arm — no multi-seed claim is supported.",
        "Historical Tesla-T4 measurements are not represented here; they live in their own",
        "labelled table and are never differenced against these A100-trained results.", "",
        "| Logical ID | Physical | Model | Total params | Trainable | Status | Best Val Top-1 | "
        "Clean Test Top-1 | Easy Test Top-1 | Moderate Test Top-1 | Hard Test Top-1 | Clean→Hard drop | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['Logical ID']} | {r['Physical run']} | {r['Model']} | {fmt(r['Total params'])} | "
            f"{fmt(r['Trainable params'])} | {r['Training status']} | {fmt(r['Best Val Top-1'],7)} | "
            f"{fmt(r[COL['clean']])} | {fmt(r[COL['easy']])} | {fmt(r[COL['moderate']])} | "
            f"{fmt(r[COL['hard']])} | {fmt(r['Robustness drop (clean -> hard)'])} | {r['Notes']} |")
    lines += ["", f"Source: `{args.eval_root}` prediction records; training facts from "
                  "`docs/RUN_INVENTORY.md`.", ""]
    if mismatches:
        lines += ["## Aggregate consistency FAILURES", ""] + [f"- {m}" for m in mismatches] + [""]
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    open(args.out_md, "w").write("\n".join(lines))

    have = sum(1 for r in rows if r[COL["clean"]] is not None)
    print(f"{len(rows)} logical IDs; {have} with evaluation results")
    print(f"wrote {args.out_csv}\nwrote {args.out_md}")
    if mismatches:
        print("\nAGGREGATE CONSISTENCY FAILURES:", file=sys.stderr)
        for m in mismatches:
            print("  " + m, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
