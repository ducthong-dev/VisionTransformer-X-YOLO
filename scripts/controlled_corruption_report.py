#!/usr/bin/env python
"""Per-family report and paired statistics for the Controlled Synthetic Corruption Benchmark.

    python scripts/controlled_corruption_report.py

Reports EVERY corruption family separately. A grand mean across families is deliberately
not the headline: averaging Gaussian noise with JPEG and brightness hides exactly the
failure modes this benchmark exists to expose. A mean is printed only as a clearly
labelled secondary column.

Per model x family x severity: Top-1, Macro-F1, absolute degradation from Clean, and
retention from Clean (dist_top1 / clean_top1).

Paired model-vs-model statistics are computed WITHIN each corruption distribution --
valid at a single training seed because both models see the identical corrupted images.
The headline comparison is A5 vs D1: the cleanest available test of whether the
denoising objective improves robustness.

Never merges this benchmark with Clean/Easy/Moderate/Hard.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_statistics import holm, macro_f1, mcnemar, paired_bootstrap  # noqa: E402

BENCHMARK = "Controlled Synthetic Corruption Benchmark"
SEVERITIES = ["mild", "moderate", "severe"]

# Frozen before results were inspected. A5 vs D1 first: the denoising objective.
COMPARISONS = [
    ("A5", "D1", "PRIMARY  denoising objective (A5 denoising vs D1 clean objective)"),
    ("A5", "A0", "Original AE-TFPE vs fair baseline"),
    ("E5", "A0", "Efficient AE-TFPE vs fair baseline"),
    ("E5", "B2", "Efficient AE-TFPE vs external lightweight baseline"),
]
DISCLAIMER = ("Paired intervals quantify TEST-SAMPLE uncertainty only. Training used a single "
              "seed (0); they do not substitute for multi-seed retraining.")


def load(root, rid, key):
    p = os.path.join(root, rid, f"predictions_{key}.csv.gz")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, usecols=["sample_id", "ground_truth_index", "predicted_index",
                                 "correct_top1", "correct_top5"])
    return df.sort_values("sample_id").reset_index(drop=True)


def metrics(df):
    gt = df["ground_truth_index"].to_numpy()
    pr = df["predicted_index"].to_numpy()
    k = int(max(gt.max(), pr.max())) + 1
    return {"top1": float(df["correct_top1"].mean()),
            "top5": float(df["correct_top5"].mean()),
            "macro_f1": macro_f1(gt, pr, k),
            "n": int(len(df))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="results/evaluation_controlled")
    ap.add_argument("--frozen", default="results/controlled_corruptions")
    ap.add_argument("--spec", default="configs/controlled_corruptions.yaml")
    ap.add_argument("--out-json", default="results/evaluation_controlled/controlled_report.json")
    ap.add_argument("--out-md", default="docs/CONTROLLED_CORRUPTION_RESULTS.md")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.spec))
    families = list(cfg["corruptions"])
    names = cfg["display_names"]

    if not os.path.isdir(args.eval_root):
        print(f"ERROR: {args.eval_root} not found -- run scripts/evaluate_controlled_corruptions.py first.",
              file=sys.stderr)
        return 2
    runs = sorted(d for d in os.listdir(args.eval_root)
                  if os.path.isdir(os.path.join(args.eval_root, d)))
    if not runs:
        print("ERROR: no evaluated runs.", file=sys.stderr)
        return 2

    report = {"benchmark": BENCHMARK,
              "kind": "targeted corruption / noise robustness (non-geometric)",
              "separate_from": "Clean/Easy/Moderate/Hard (synthetic augmentation robustness)",
              "never_merged": True, "disclaimer": DISCLAIMER,
              "seed_variability": "UNAVAILABLE (single training seed)",
              "per_model": {}, "paired": {}}

    # ---- per-model, per-family, per-severity -------------------------------- #
    for rid in runs:
        clean = load(args.eval_root, rid, "clean_none")
        if clean is None:
            continue
        cm = metrics(clean)
        entry = {"clean": cm, "families": {}}
        for fam in families:
            per_sev = {}
            for sev in SEVERITIES:
                df = load(args.eval_root, rid, f"{fam}_{sev}")
                if df is None:
                    continue
                m = metrics(df)
                m["degradation_top1"] = round(cm["top1"] - m["top1"], 6)
                m["retention_top1"] = round(m["top1"] / cm["top1"], 6) if cm["top1"] else None
                m["degradation_macro_f1"] = round(cm["macro_f1"] - m["macro_f1"], 6)
                per_sev[sev] = m
            if per_sev:
                vals = [v["top1"] for v in per_sev.values()]
                entry["families"][fam] = {
                    "display_name": names.get(fam, fam), "severities": per_sev,
                    "mean_top1_over_severities": round(float(np.mean(vals)), 6),
                }
        report["per_model"][rid] = entry

    # ---- paired model-vs-model within each corruption distribution ---------- #
    dists = ["clean_none"] + [f"{f}_{s}" for f in families for s in SEVERITIES]
    for key in dists:
        rows, pvals = [], []
        for a, b, label in COMPARISONS:
            da, db = load(args.eval_root, a, key), load(args.eval_root, b, key)
            if da is None or db is None:
                continue
            if not da["sample_id"].equals(db["sample_id"]):
                print(f"  {key}: {a} vs {b} sample sets differ, SKIPPED", file=sys.stderr)
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
            report["paired"][key] = rows

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump(report, open(args.out_json, "w"), indent=2)

    # ---- markdown ----------------------------------------------------------- #
    L = [f"# {BENCHMARK} — Results", "",
         "Targeted **corruption / noise robustness** on 6 non-geometric families.",
         "Reported **separately** from Clean/Easy/Moderate/Hard (synthetic *augmentation*",
         "robustness); the two are never merged into a single average.", "",
         f"Single training seed (0). {DISCLAIMER}", ""]

    def f4(v):
        return "—" if v is None else f"{v:.4f}"

    for sev in SEVERITIES:
        L += [f"## Top-1 by family — severity: {sev}", "",
              "| Model | Clean | " + " | ".join(names.get(f, f) for f in families) +
              " | mean over families* |",
              "|---" * (len(families) + 3) + "|"]
        for rid, e in report["per_model"].items():
            cells = []
            for fam in families:
                s = e["families"].get(fam, {}).get("severities", {}).get(sev)
                cells.append(f4(s["top1"]) if s else "—")
            got = [float(c) for c in cells if c != "—"]
            L.append(f"| {rid} | {f4(e['clean']['top1'])} | " + " | ".join(cells) +
                     f" | {f4(float(np.mean(got)) if got else None)} |")
        L += ["", "\\* secondary only — a mean across families hides family-specific failure modes.", ""]

    L += ["## Degradation and retention from Clean (severity: severe)", "",
          "| Model | Family | Clean Top-1 | Corrupted Top-1 | Abs. degradation | Retention | Macro-F1 drop |",
          "|---|---|---|---|---|---|---|"]
    for rid, e in report["per_model"].items():
        for fam in families:
            s = e["families"].get(fam, {}).get("severities", {}).get("severe")
            if s:
                L.append(f"| {rid} | {names.get(fam, fam)} | {f4(e['clean']['top1'])} | "
                         f"{f4(s['top1'])} | {f4(s['degradation_top1'])} | "
                         f"{f4(s['retention_top1'])} | {f4(s['degradation_macro_f1'])} |")
    L.append("")

    L += ["## Paired model-vs-model statistics", "",
          "Within each corruption distribution both models see identical images, so these",
          "comparisons are paired and valid at one seed. Holm-corrected across the",
          "comparison set within each distribution.", ""]
    for key, rows in report["paired"].items():
        L += [f"### {key}", "",
              "| Comparison | A Top-1 | B Top-1 | Δ | 95% CI | McNemar p | Holm p | Detectable |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            d = r["bootstrap"]["accuracy"]
            L.append(f"| {r['a']} vs {r['b']} — {r['comparison']} | {f4(r['a_top1'])} | "
                     f"{f4(r['b_top1'])} | {d['observed_difference']:+.4f} | "
                     f"[{d['ci95_low']:+.4f}, {d['ci95_high']:+.4f}] | "
                     f"{r['mcnemar']['p_value']:.3g} | {r['mcnemar']['p_value_holm']:.3g} | "
                     f"{'yes' if r['detectable'] else 'no'} |")
        L.append("")

    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    open(args.out_md, "w").write("\n".join(L))
    print(f"models reported: {len(report['per_model'])}")
    print(f"wrote {args.out_json}\nwrote {args.out_md}")
    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
