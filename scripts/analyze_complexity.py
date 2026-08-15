#!/usr/bin/env python
"""Parameters / FLOPs / model size / latency / throughput for every arm (Phase 8).

    python scripts/analyze_complexity.py --device mps --batch-size 1

All arms are measured in one process on one device at one batch size and
resolution, with fixed warm-up and timed iteration counts, so the numbers are
comparable by construction. The device and host are recorded in the output.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.complexity import (  # noqa: E402
    count_flops, count_parameters, hardware_info, measure_latency, model_size_mb,
)
from aetfpe.config import load_experiment, pick_device, resolve_roots  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.seeding import seed_everything  # noqa: E402

FIELDS = [
    "name", "group", "params_total", "params_trainable", "params_frozen",
    "params_frontend", "model_size_mb", "img_size", "gflops", "latency_ms_mean",
    "latency_ms_std", "latency_ms_per_image", "throughput_img_per_s", "device",
]

# FLOPs are reported at the TRAINING resolution (224), counting 2 x MACs.
# Ultralytics' own `model.info()` reports YOLOv8n-cls as "3.4 GFLOPs", but that
# is measured at its default 640x640; 640^2 / 224^2 = 8.16, and
# 0.4116 x 8.16 = 3.36. The two numbers agree once the resolution is stated.
# Always report the resolution alongside the figure.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--num-classes", type=int, default=39)
    ap.add_argument("--out", default=None, help="default: ${OUTPUT_ROOT}/complexity")
    ap.add_argument("--skip-latency", action="store_true",
                    help="parameters and FLOPs only. Use on a laptop: those are "
                         "hardware-independent, latency is not.")
    args = ap.parse_args()

    paths = args.configs or sorted(
        p for p in glob.glob("configs/*.yaml")
        if os.path.basename(p) not in ("_base.yaml", "corruptions.yaml")
        and not os.path.basename(p).startswith("local")
    )
    device = pick_device(args.device)
    args.out = args.out or os.path.join(resolve_roots()["OUTPUT_ROOT"], "complexity")
    os.makedirs(args.out, exist_ok=True)
    seed_everything(0)

    rows = []
    for p in paths:
        cfg = load_experiment(p)
        cfg.setdefault("model", {})["num_classes"] = args.num_classes
        name = cfg.get("name", os.path.basename(p))
        model = build_model(cfg["model"]).eval()

        row = {"name": name, "group": cfg.get("group", "misc"), "img_size": args.img_size}
        row.update(count_parameters(model))
        row["params_frontend"] = int(sum(q.numel() for q in model.frontend_parameters()))
        row["model_size_mb"] = round(model_size_mb(model), 3)
        # FLOPs on CPU: thop's hooks are device-agnostic and CPU avoids MPS tracing gaps
        row.update(count_flops(model, (1, 3, args.img_size, args.img_size), device="cpu"))
        row["gflops"] = round(row.get("gflops", float("nan")), 4)

        if args.skip_latency:
            for k in ("latency_ms_mean", "latency_ms_std", "latency_ms_per_image",
                      "throughput_img_per_s"):
                row[k] = float("nan")
            row["device"] = "not measured"
            print(f"{name:<24} params={row['params_total']:>12,} gflops={row['gflops']:>8} "
                  f"(latency skipped)")
        else:
            row.update(
                measure_latency(model, device=device, batch_size=args.batch_size,
                                img_size=args.img_size, warmup=args.warmup, iters=args.iters)
            )
            for k in ("latency_ms_mean", "latency_ms_std", "latency_ms_per_image",
                      "throughput_img_per_s"):
                row[k] = round(row[k], 4)
            print(f"{name:<24} params={row['params_total']:>12,} "
                  f"gflops={row['gflops']:>8} "
                  f"lat={row['latency_ms_mean']:>8.2f}+-{row['latency_ms_std']:.2f} ms "
                  f"thr={row['throughput_img_per_s']:.1f} img/s")
        rows.append(row)

    with open(os.path.join(args.out, "complexity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    hw = hardware_info("not measured" if args.skip_latency else device)
    with open(os.path.join(args.out, "complexity.json"), "w") as fh:
        json.dump({
            "hardware": hw,
            "batch_size": args.batch_size, "img_size": args.img_size,
            "warmup_iters": args.warmup, "timed_iters": args.iters,
            "latency_measured": not args.skip_latency,
            "rows": rows,
        }, fh, indent=2)

    print(f"\nwrote {len(rows)} rows -> {args.out}/complexity.csv")
    if not hw.get("timings_reportable", False) and not args.skip_latency:
        print(f"\n  !! {hw['timing_note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
