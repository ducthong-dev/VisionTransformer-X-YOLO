#!/usr/bin/env python
"""Architecture v2 benchmark -- inference complexity of the four candidates.

EXPLORATORY. Trains nothing, loads no checkpoint, touches no dataset. Builds each
candidate with randomly-initialised (or ImageNet-pretrained) weights and measures
inference cost only.

    LOCAL   python scripts/benchmark_architectures.py --device cpu
    COLAB   python scripts/benchmark_architectures.py --device cuda    # <- the real run

Benchmark protocol, identical for every candidate including the baseline:
    batch sizes 1 and 32 | 50 warm-up iters | 200 timed iters
    torch.cuda.synchronize() around every timed region
    AMP setting identical across candidates (default: off, matching training)

Parameters, FLOPs/MACs and tensor shapes are hardware-independent and are valid
from any device. Latency, throughput and peak memory are CUDA-only: on a non-CUDA
device they are recorded but stamped `timings_reportable: false`, per
SCIENTIFIC_PROTOCOL_FROZEN.md.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.complexity import count_flops, count_parameters, hardware_info, model_size_mb  # noqa: E402
from aetfpe.config import environment_info, pick_device, resolve_roots  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.seeding import seed_everything  # noqa: E402

# The four candidates. C0 is the frozen A5_aetfpe_full, reproduced here exactly so
# every other row is measured against it under one identical protocol.
CANDIDATES = {
    "BASELINE": {
        "label": "YOLOv8n-cls only (reference)",
        "model": dict(use_pe=False, use_tf=False, use_ae=False, fusion="identity"),
    },
    "C0": {
        "label": "ViT-B/16 + image-space AE + YOLOv8n-cls  (current A5, frozen v1)",
        "model": dict(use_pe=True, use_tf=True, use_ae=True, fusion="linear"),
    },
    "C1": {
        "label": "MobileViT-XXS + image-space AE + YOLOv8n-cls",
        "model": dict(use_pe=True, use_tf=True, use_ae=True, fusion="linear",
                      tf_backbone="mobilevit_xxs", ae_space="image"),
    },
    "C2": {
        "label": "MobileViT-XXS + slim feature-space AE + YOLOv8n-cls",
        "model": dict(use_pe=True, use_tf=True, use_ae=True, fusion="linear",
                      tf_backbone="mobilevit_xxs", ae_space="feature"),
    },
    "C3": {
        "label": "EfficientViT-B0 + slim feature-space AE + YOLOv8n-cls",
        "model": dict(use_pe=True, use_tf=True, use_ae=True, fusion="linear",
                      tf_backbone="efficientvit_b0", ae_space="feature"),
    },
}

# Frozen decision rules (docs/ARCHITECTURE_V2_BENCHMARK.md). Evaluated, never tuned.
PREFERRED_PARAMS = 3_000_000
PREFERRED_FLOPS_RATIO = 3.0
PREFERRED_LATENCY_RATIO = 3.0
REJECT_FLOPS_RATIO = 5.0
REJECT_LATENCY_RATIO = 5.0


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


@torch.no_grad()
def measure_latency(model, device, batch_size, img_size, warmup, iters, amp):
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (amp and device.startswith("cuda"))
        else torch.autocast(device_type="cpu", enabled=False)
    )

    for _ in range(warmup):
        with autocast:
            model(x)
    sync(device)

    times = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        with autocast:
            model(x)
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    mean = statistics.fmean(times)
    return {
        "batch_size": batch_size,
        "latency_ms_mean": round(mean, 4),
        "latency_ms_std": round(statistics.pstdev(times), 4),
        "latency_ms_median": round(statistics.median(times), 4),
        "latency_ms_per_image": round(mean / batch_size, 5),
        "throughput_img_per_s": round(batch_size * 1000.0 / mean, 2),
        "warmup_iters": warmup,
        "timed_iters": iters,
    }


@torch.no_grad()
def peak_memory(model, device, batch_size, img_size):
    if not device.startswith("cuda"):
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    model(x)
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)


def interface_description(model) -> dict:
    """The exact dimensional interface between transformer, AE and YOLO."""
    d = {}
    tf = getattr(model, "tf", None)
    if tf is None:
        d["transformer"] = None
    else:
        desc = tf.describe()
        d["transformer"] = {
            "name": desc.get("model_name"),
            "library": desc.get("library"),
            "frozen": desc.get("frozen"),
            "grid_channels": desc.get("backbone_channels", desc.get("embedding_dim")),
        }
    ae = getattr(model, "ae", None)
    d["autoencoder"] = ae.describe() if ae is not None else None
    d["ae_space"] = model.cfg.ae_space if model.cfg.use_ae else None
    d["classifier_in_channels"] = model.classifier_in_channels
    d["classifier_stem_modified"] = model.classifier_in_channels != 3
    return d


@torch.no_grad()
def trace_shapes(model, device, img_size) -> dict:
    # rand, not randn: the dataloader delivers images in [0, 1], and the reported
    # classifier-input range is only meaningful in that domain. (Latency below
    # uses randn, which is fine -- timing does not depend on the input's range.)
    x = torch.rand(2, 3, img_size, img_size, device=device)
    out, parts = model.frontend(x, return_parts=True)
    logits = model(x)
    return {
        "input": list(x.shape),
        "fused_pre_ae": list(parts["pre_ae"].shape) if parts["pre_ae"] is not None else None,
        "ae_latent": list(parts["latent"].shape) if parts["latent"] is not None else None,
        "classifier_input": list(out.shape),
        "logits": list(logits.shape),
        "classifier_input_range": [round(float(out.min()), 4), round(float(out.max()), 4)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-classes", type=int, default=39)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--amp", action="store_true", help="identical across candidates; default off")
    ap.add_argument("--pretrained", action="store_true",
                    help="download pretrained encoder weights (irrelevant to cost)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    reportable = device.startswith("cuda")
    args.out = args.out or os.path.join(resolve_roots()["OUTPUT_ROOT"], "architecture_v2",
                                        "benchmark.json")
    seed_everything(0)

    keys = args.only or list(CANDIDATES)
    print(f"device={device}  amp={args.amp}  img={args.img_size}  "
          f"warmup={args.warmup}  iters={args.iters}")
    print(f"latency/memory reportable: {reportable}"
          f"{'' if reportable else '  <- NON-CUDA: cost figures valid, timings are NOT'}\n")

    rows = []
    for key in keys:
        spec = CANDIDATES[key]
        cfg = dict(spec["model"])
        cfg["num_classes"] = args.num_classes
        cfg["img_size"] = args.img_size
        cfg["pretrained"] = True
        cfg["vit_pretrained"] = bool(args.pretrained)

        model = build_model(cfg).to(device).eval()

        row = {"candidate": key, "label": spec["label"], "config": spec["model"]}
        row.update(count_parameters(model))
        row["model_size_mb"] = round(model_size_mb(model), 3)
        row["img_size"] = args.img_size
        row.update(count_flops(model, (1, 3, args.img_size, args.img_size), device="cpu"))
        row["gflops"] = round(row.get("gflops", float("nan")), 4)
        row["macs_g"] = round(row.get("macs", float("nan")) / 1e9, 4)
        row["shapes"] = trace_shapes(model, device, args.img_size)
        row["interface"] = interface_description(model)

        row["latency"] = {}
        for bs in args.batch_sizes:
            row["latency"][str(bs)] = measure_latency(
                model, device, bs, args.img_size, args.warmup, args.iters, args.amp)
            pm = peak_memory(model, device, bs, args.img_size)
            if pm is not None:
                row["latency"][str(bs)]["peak_gpu_mem_mb"] = pm

        rows.append(row)
        l1 = row["latency"][str(args.batch_sizes[0])]
        print(f"{key:<9} {row['params_total']:>11,} params  {row['gflops']:>8.4f} GFLOPs  "
              f"bs{args.batch_sizes[0]} {l1['latency_ms_mean']:>8.3f} ms  "
              f"{l1['throughput_img_per_s']:>8.1f} img/s")
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    base = next((r for r in rows if r["candidate"] == "BASELINE"), None)
    if base:
        for r in rows:
            r["vs_baseline"] = {
                "params_ratio": round(r["params_total"] / base["params_total"], 3),
                "gflops_ratio": round(r["gflops"] / base["gflops"], 3),
            }
            for bs in args.batch_sizes:
                b = base["latency"][str(bs)]["latency_ms_mean"]
                c = r["latency"][str(bs)]["latency_ms_mean"]
                r["vs_baseline"][f"latency_ratio_bs{bs}"] = round(c / b, 3)

            fl = r["vs_baseline"]["gflops_ratio"]
            lat = r["vs_baseline"][f"latency_ratio_bs{args.batch_sizes[0]}"]
            hard_reject = fl > REJECT_FLOPS_RATIO or (reportable and lat > REJECT_LATENCY_RATIO)
            preferred = (
                r["params_total"] <= PREFERRED_PARAMS
                and fl <= PREFERRED_FLOPS_RATIO
                and (not reportable or lat <= PREFERRED_LATENCY_RATIO)
            )
            r["verdict"] = {
                "hard_reject": bool(hard_reject),
                "meets_preferred": bool(preferred),
                "latency_criterion_evaluated": reportable,
            }

    payload = {
        "device": device,
        "timings_reportable": reportable,
        "amp": args.amp,
        "img_size": args.img_size,
        "batch_sizes": args.batch_sizes,
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "decision_rules": {
            "preferred_params_max": PREFERRED_PARAMS,
            "preferred_gflops_ratio_max": PREFERRED_FLOPS_RATIO,
            "preferred_latency_ratio_max": PREFERRED_LATENCY_RATIO,
            "reject_gflops_ratio_above": REJECT_FLOPS_RATIO,
            "reject_latency_ratio_above": REJECT_LATENCY_RATIO,
        },
        "hardware": hardware_info(device),
        "environment": environment_info(),
        "candidates": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"\nwrote {args.out}")
    if not reportable:
        print("NOTE: parameters / FLOPs / shapes are hardware-independent and final.\n"
              "      Latency, throughput and peak memory require --device cuda on a T4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
