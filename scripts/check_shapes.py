#!/usr/bin/env python
"""Tensor-shape smoke test for every arm (part of the Phase-1 validation gate).

Builds each config, pushes a batch through the front-end and the classifier, and
prints the shape at every stage plus the parameter split. Run this before any
training; it catches interface errors in seconds rather than at epoch 1.

    python scripts/check_shapes.py
    python scripts/check_shapes.py --configs configs/aetfpe_full.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.complexity import count_parameters  # noqa: E402
from aetfpe.config import load_experiment, resolve_roots  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.models.classifier import classifier_forward  # noqa: E402
from aetfpe.seeding import seed_everything  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num-classes", type=int, default=39)
    ap.add_argument("--out", default=None, help="default: ${OUTPUT_ROOT}/validation/shape_report.json")
    args = ap.parse_args()

    paths = args.configs or sorted(
        p for p in glob.glob("configs/*.yaml")
        if os.path.basename(p) not in ("_base.yaml", "corruptions.yaml")
        and not os.path.basename(p).startswith("local")
    )

    args.out = args.out or os.path.join(
        resolve_roots()["OUTPUT_ROOT"], "validation", "shape_report.json")
    seed_everything(0)
    x = torch.rand(args.batch, 3, 224, 224)
    report, failures = [], 0

    for p in paths:
        cfg = load_experiment(p)
        cfg.setdefault("model", {})["num_classes"] = args.num_classes
        name = cfg.get("name", os.path.basename(p))
        try:
            model = build_model(cfg["model"]).eval()
            with torch.no_grad():
                out, parts = model.frontend(x, return_parts=True)
                logits = classifier_forward(model.classifier, out)

            counts = count_parameters(model)
            fe = sum(q.numel() for q in model.frontend_parameters())
            row = {
                "name": name,
                "config": p,
                "input": list(x.shape),
                "pre_ae": list(parts["pre_ae"].shape) if parts["pre_ae"] is not None else None,
                "latent": list(parts["latent"].shape) if parts["latent"] is not None else None,
                "classifier_input": list(out.shape),
                "logits": list(logits.shape),
                "classifier_in_channels": model.classifier_in_channels,
                "stem_modified": model.classifier_in_channels != 3,
                "params_total": counts["params_total"],
                "params_trainable": counts["params_trainable"],
                "params_frontend": int(fe),
                "output_range": [round(float(out.min()), 4), round(float(out.max()), 4)],
                "ok": list(logits.shape) == [args.batch, args.num_classes],
            }
            if not row["ok"]:
                failures += 1
            report.append(row)
            print(f"{name:<24} pre_ae={str(row['pre_ae']):<22} latent={str(row['latent']):<20} "
                  f"cls_in={str(row['classifier_input']):<20} logits={row['logits']} "
                  f"params={counts['params_total']:,} {'OK' if row['ok'] else 'FAIL'}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            report.append({"name": name, "config": p, "ok": False, "error": repr(exc)})
            print(f"{name:<24} FAILED: {exc!r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n{len(report) - failures}/{len(report)} arms passed -> {args.out}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
