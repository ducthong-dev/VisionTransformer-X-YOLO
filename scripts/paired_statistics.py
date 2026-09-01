#!/usr/bin/env python
"""Paired test-sample statistics over the frozen prediction records.

    python scripts/paired_statistics.py
    python scripts/paired_statistics.py --distributions clean hard

Two sources of uncertainty exist and are never conflated:

  A. TRAINING-SEED VARIABILITY -- UNAVAILABLE. Every arm was trained once, at seed 0.
     Nothing here estimates it, and no output may be read as a multi-seed claim.

  B. PAIRED TEST-SAMPLE UNCERTAINTY -- what this script measures. Two models see the
     same 8335 images, so a model-vs-model comparison WITHIN one distribution is paired
     and is valid at a single seed. Cross-distribution comparisons for one model are
     also paired, because the clean<->augmented mapping is PROVEN.

Tests: McNemar (exact binomial on discordant pairs) for top-1 correctness, and a paired
bootstrap CI for differences in accuracy and macro-F1. Holm-Bonferroni within each
distribution; raw and adjusted p-values are both reported.

The comparison list is the one frozen in docs/GATE1_RESOLUTION_AND_ANALYSIS_PROTOCOL.md
before any result was seen. Anything added later must be labelled post-hoc.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# Frozen before results were inspected. (id_a, id_b, question)
COMPARISONS = [
    ("E5", "A0", "C1  Efficient AE-TFPE vs fair baseline"),
    ("E5", "M1", "C2  vs legacy-LUT control"),
    ("E5", "M2", "C2  vs photometric (gamma) control"),
    ("E5", "M3", "C2  vs augmentation control"),
    ("E5", "E3", "C3  AE contribution, Efficient side [CONFOUNDED: also changes fusion space]"),
    ("E5", "E7", "C4  28x28 vs 7x7 grid"),
    ("A5", "D1", "C5  denoising objective -- the cleanest available test"),
    ("A5", "A3", "C6  AE contribution, Original side"),
    ("A5", "F1", "C7  AE fusion vs addition"),
    ("A5", "F2", "C7  AE fusion vs concatenation [CONFOUNDED: F2 modifies classifier stem]"),
    ("A5", "F4", "C7  AE fusion vs attention"),
    ("A5", "E5", "C8  Original vs Efficient formulation"),
    ("A0", "B2", "C9  our baseline vs external lightweight baseline"),
]
DISTRIBUTIONS = ["clean", "easy", "moderate", "hard"]
BOOTSTRAP = 10000
SEED = 0
DISCLAIMER = ("Quantifies TEST-SAMPLE uncertainty only. Training used a single seed (0); "
              "these intervals do NOT substitute for multi-seed retraining and do not "
              "establish architectural superiority.")


def load(run_out: str, rid: str, dist: str) -> pd.DataFrame | None:
    p = os.path.join(run_out, rid, f"predictions_{dist}.csv.gz")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, usecols=["sample_id", "ground_truth_index", "predicted_index",
                                 "correct_top1", "correct_top5"])
    return df.sort_values("sample_id").reset_index(drop=True)


def macro_f1(gt: np.ndarray, pred: np.ndarray, k: int) -> float:
    f1 = []
    for c in range(k):
        tp = int(((pred == c) & (gt == c)).sum())
        fp = int(((pred == c) & (gt != c)).sum())
        fn = int(((pred != c) & (gt == c)).sum())
        d = 2 * tp + fp + fn
        f1.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(f1))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """a, b: boolean top-1 correctness, same samples, same order."""
    b01 = int(((a == 1) & (b == 0)).sum())      # a right, b wrong
    b10 = int(((a == 0) & (b == 1)).sum())      # a wrong, b right
    n = b01 + b10
    p = 1.0 if n == 0 else binomtest(b01, n, 0.5, alternative="two-sided").pvalue
    return {"a_only_correct": b01, "b_only_correct": b10, "discordant": n, "p_value": float(p)}


def paired_bootstrap(gt, pa, pb, k, n=BOOTSTRAP, seed=SEED) -> dict:
    rng = np.random.default_rng(seed)
    m = len(gt)
    ca, cb = (pa == gt), (pb == gt)
    d_acc = np.empty(n)
    d_f1 = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)             # SAME indices for both models -> paired
        d_acc[i] = ca[idx].mean() - cb[idx].mean()
        d_f1[i] = macro_f1(gt[idx], pa[idx], k) - macro_f1(gt[idx], pb[idx], k)
    out = {}
    for name, arr, obs in (("accuracy", d_acc, float(ca.mean() - cb.mean())),
                           ("macro_f1", d_f1, macro_f1(gt, pa, k) - macro_f1(gt, pb, k))):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[name] = {"observed_difference": round(obs, 6),
                     "ci95_low": round(float(lo), 6), "ci95_high": round(float(hi), 6),
                     "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="results/evaluation")
    ap.add_argument("--out", default="results/evaluation/paired_statistics.json")
    ap.add_argument("--distributions", nargs="*", default=DISTRIBUTIONS)
    ap.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = ap.parse_args()

    if not os.path.isdir(args.eval_root):
        print(f"ERROR: {args.eval_root} does not exist -- run scripts/evaluate_distributions.py first.",
              file=sys.stderr)
        return 2

    report = {"disclaimer": DISCLAIMER, "bootstrap_resamples": args.bootstrap,
              "bootstrap_seed": SEED, "seed_variability": "UNAVAILABLE (single training seed)",
              "within_distribution": {}, "cross_distribution": {}}

    for dist in args.distributions:
        rows, pvals = [], []
        for a, b, label in COMPARISONS:
            da, db = load(args.eval_root, a, dist), load(args.eval_root, b, dist)
            if da is None or db is None:
                continue
            if not da["sample_id"].equals(db["sample_id"]):
                print(f"  {dist}: {a} vs {b} -- sample sets differ, SKIPPED", file=sys.stderr)
                continue
            gt = da["ground_truth_index"].to_numpy()
            pa, pb = da["predicted_index"].to_numpy(), db["predicted_index"].to_numpy()
            k = int(max(gt.max(), pa.max(), pb.max())) + 1
            mc = mcnemar((pa == gt).astype(int), (pb == gt).astype(int))
            bs = paired_bootstrap(gt, pa, pb, k, n=args.bootstrap)
            rows.append({"comparison": label, "a": a, "b": b,
                         "a_top1": round(float((pa == gt).mean()), 6),
                         "b_top1": round(float((pb == gt).mean()), 6),
                         "mcnemar": mc, "bootstrap": bs, "n_samples": int(len(gt))})
            pvals.append(mc["p_value"])
        if rows:
            for r, adj in zip(rows, holm(pvals)):
                r["mcnemar"]["p_value_holm"] = round(float(adj), 6)
                r["detectable"] = bool(r["bootstrap"]["accuracy"]["excludes_zero"] and adj < 0.05)
            report["within_distribution"][dist] = rows
            print(f"\n== {dist} ==")
            for r in rows:
                d = r["bootstrap"]["accuracy"]
                flag = "DETECTABLE" if r["detectable"] else "not detectable"
                print(f"  {r['a']:>3s} vs {r['b']:<3s} {r['a_top1']:.4f} vs {r['b_top1']:.4f}  "
                      f"d={d['observed_difference']:+.4f} [{d['ci95_low']:+.4f},{d['ci95_high']:+.4f}]  "
                      f"p={r['mcnemar']['p_value']:.3g} holm={r['mcnemar']['p_value_holm']:.3g}  {flag}")

    # ---- cross-distribution degradation, per model (mapping is PROVEN) ------- #
    runs = sorted(d for d in os.listdir(args.eval_root)
                  if os.path.isdir(os.path.join(args.eval_root, d)))
    for rid in runs:
        base = load(args.eval_root, rid, "clean")
        if base is None:
            continue
        entry = {}
        for dist in [d for d in args.distributions if d != "clean"]:
            cur = load(args.eval_root, rid, dist)
            if cur is None or not base["sample_id"].equals(cur["sample_id"]):
                continue
            mc = mcnemar(base["correct_top1"].to_numpy(), cur["correct_top1"].to_numpy())
            entry[dist] = {"clean_top1": round(float(base["correct_top1"].mean()), 6),
                           "dist_top1": round(float(cur["correct_top1"].mean()), 6),
                           "drop": round(float(base["correct_top1"].mean() - cur["correct_top1"].mean()), 6),
                           "mcnemar": mc}
        if entry:
            report["cross_distribution"][rid] = entry

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
