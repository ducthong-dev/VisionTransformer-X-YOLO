#!/usr/bin/env python
"""Confusion matrix and per-class report (Phase 9).

Reads the JSON that scripts/evaluate.py already wrote, so it never re-runs the
model and can never disagree with the reported accuracy.

    python scripts/confusion_matrix.py --run results/ablation/A0_baseline_rgb
    python scripts/confusion_matrix.py --run results/... --condition pepper_030
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def load_condition(run: str, condition: str) -> dict:
    if condition in ("clean", "clean_testsplit"):
        path = os.path.join(run, "test_clean.json")
    else:
        path = os.path.join(run, "per_class", f"{condition}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no results at {path}; run scripts/evaluate.py first")
    with open(path) as fh:
        return json.load(fh)


def plot(cm: np.ndarray, classes: list[str], title: str, out_png: str, normalize: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = cm.astype(np.float64)
    if normalize:
        row = m.sum(axis=1, keepdims=True)
        m = np.divide(m, row, out=np.zeros_like(m), where=row > 0)

    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.28), max(7, n * 0.26)), dpi=150)
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1 if normalize else None)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=90, fontsize=5)
    ax.set_yticklabels(classes, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.036, label="row-normalised rate" if normalize else "count")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--condition", default="clean",
                    help="'clean' or a corruption tag such as pepper_030")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    res = load_condition(args.run, args.condition)
    classes = [r["class"] for r in res["per_class"]]
    cm = np.array(res["confusion_matrix"], dtype=np.int64)

    out_dir = args.out or os.path.join(args.run, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.condition

    with open(os.path.join(out_dir, f"confusion_matrix_{tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + classes)
        for i, c in enumerate(classes):
            w.writerow([c] + cm[i].tolist())

    rows = sorted(res["per_class"], key=lambda r: r["f1"])
    with open(os.path.join(out_dir, f"per_class_{tag}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(res["per_class"])

    print(f"condition       : {tag}")
    print(f"top-1 / top-5   : {res['overall']['top1']:.4f} / {res['overall']['top5']:.4f}")
    print(f"macro P/R/F1    : {res['overall'].get('macro_precision', float('nan')):.4f} / "
          f"{res['overall'].get('macro_recall', float('nan')):.4f} / "
          f"{res['overall'].get('macro_f1', float('nan')):.4f}")
    print("\nweakest 5 classes by F1:")
    for r in rows[:5]:
        print(f"  {r['class']:<46} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} n={r['support']}")

    if not args.no_plot:
        png = os.path.join(out_dir, f"confusion_matrix_{tag}.png")
        plot(cm, classes, f"{os.path.basename(args.run)} - {tag}", png)
        print(f"\nplot -> {png}")

    print(f"tables -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
