#!/usr/bin/env python
"""G1 -- architecture validation: forward pass, backward pass, NaN/Inf, and
parameter-count fidelity, for every frozen arm.

This is the command named in COLAB_CAMPAIGN_PLAN.md's G1 gate. It is
device-aware (`--device cuda|mps|cpu|auto`) so the identical script is the CPU/MPS
development check AND the official CUDA gate -- only the device differs. G1 is
only formally satisfied by a run with `--device cuda`; a CPU/MPS pass is a
development sanity check, not a substitute, and the report says so explicitly.

    LOCAL   python scripts/check_shapes.py                    # CPU/MPS dev check
    COLAB   python scripts/check_shapes.py --device cuda       # official G1

Per arm, verifies:
  1. forward pass succeeds and produces the expected logit shape
  2. classifier input is in [0,1] for every arm (this is the fairness property
     found and fixed during Phase-1 validation)
  3. backward pass succeeds: loss.backward() reaches every trainable parameter
     with a finite (non-NaN, non-Inf) gradient -- including a check that NO
     trainable parameter is silently disconnected from the loss (a "built but
     never called" module is treated as a failure, not a warning)
  4. no NaN/Inf anywhere in the forward activations
  5. for YOLOv8n-cls arms with zero front-end parameters (A0/A1/M1/M2/M3), the
     parameter count matches the RECOVERED historical value in
     log-org-280223 (1,488,247) exactly. This check does not apply to the
     external baselines (ResNet-50/EfficientNet-B0/ViT-B/16), which have no
     recovered historical parameter count to compare against.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.complexity import count_parameters  # noqa: E402
from aetfpe.config import load_experiment, pick_device, resolve_roots  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.models.classifier import classifier_forward  # noqa: E402
from aetfpe.seeding import seed_everything  # noqa: E402

# RECOVERED: log-org-280223 reports YOLOv8n-cls at exactly 1,488,247 parameters
# with pretrained=True, "Transferred 156/158 items". Any arm that is the stock
# classifier with no front-end (front-end params == 0) must match this exactly.
EXPECTED_STOCK_YOLO_PARAMS = 1_488_247


def has_nan_or_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--num-classes", type=int, default=39)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="default: ${OUTPUT_ROOT}/validation/shape_report.json")
    args = ap.parse_args()

    paths = args.configs or sorted(
        p for p in glob.glob("configs/*.yaml")
        if os.path.basename(p) not in ("_base.yaml", "corruptions.yaml")
        and not os.path.basename(p).startswith("local")
    )

    device = pick_device(args.device)
    is_official_g1 = device.startswith("cuda")

    args.out = args.out or os.path.join(
        resolve_roots()["OUTPUT_ROOT"], "validation", "shape_report.json")
    seed_everything(0)
    x = torch.rand(args.batch, 3, 224, 224, device=device)
    y = torch.randint(0, args.num_classes, (args.batch,), device=device)
    ce = nn.CrossEntropyLoss()

    report, failures = [], 0
    print(f"device: {device}  "
          f"({'OFFICIAL G1' if is_official_g1 else 'development sanity check, NOT official G1'})\n")

    for p in paths:
        cfg = load_experiment(p)
        cfg.setdefault("model", {})["num_classes"] = args.num_classes
        name = cfg.get("name", os.path.basename(p))
        try:
            model = build_model(cfg["model"]).to(device)
            model.train()  # backward pass needs training-mode BN with batch>1

            out, parts = model.frontend(x, return_parts=True)
            logits = classifier_forward(model.classifier, out)

            fwd_nan = has_nan_or_inf(out) or has_nan_or_inf(logits)
            if parts["latent"] is not None:
                fwd_nan = fwd_nan or has_nan_or_inf(parts["latent"])

            loss = ce(logits, y)
            model.zero_grad(set_to_none=True)
            loss.backward()

            trainable = [q for q in model.parameters() if q.requires_grad]
            no_grad = [i for i, q in enumerate(trainable) if q.grad is None]
            bad_grad = [i for i, q in enumerate(trainable)
                       if q.grad is not None and has_nan_or_inf(q.grad)]
            backward_ok = not no_grad and not bad_grad

            counts = count_parameters(model)
            fe = sum(q.numel() for q in model.frontend_parameters())
            is_stock_yolo = cfg["model"].get("classifier", "yolov8n-cls").lower().startswith("yolo")
            expected_ok = True
            if fe == 0 and is_stock_yolo:  # stock YOLO, no front-end -> historical count applies
                expected_ok = counts["params_total"] == EXPECTED_STOCK_YOLO_PARAMS

            classifier_in_ok = bool(out.min() >= -1e-4 and out.max() <= 1 + 1e-4)

            ok = (
                list(logits.shape) == [args.batch, args.num_classes]
                and not fwd_nan and backward_ok and expected_ok and classifier_in_ok
            )

            row = {
                "name": name, "config": p, "device": device,
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
                "forward_finite": not fwd_nan,
                "backward_ok": backward_ok,
                "params_missing_grad": len(no_grad),
                "params_nonfinite_grad": len(bad_grad),
                "expected_param_count_ok": expected_ok,
                "classifier_input_range_ok": classifier_in_ok,
                "ok": bool(ok),
            }
            if not ok:
                failures += 1
            report.append(row)
            flag = "OK" if ok else "FAIL"
            print(f"{name:<24} logits={row['logits']} params={counts['params_total']:>11,} "
                  f"fwd_finite={not fwd_nan} bwd_ok={backward_ok} "
                  f"expected_params={expected_ok} range_ok={classifier_in_ok}  {flag}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            report.append({"name": name, "config": p, "device": device, "ok": False, "error": repr(exc)})
            print(f"{name:<24} FAILED: {exc!r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({
            "device": device, "is_official_g1": is_official_g1,
            "expected_stock_yolo_params": EXPECTED_STOCK_YOLO_PARAMS,
            "arms": report,
        }, fh, indent=2)

    print(f"\n{len(report) - failures}/{len(report)} arms passed -> {args.out}")
    if not is_official_g1:
        print(f"NOTE: this run used device={device!r}. G1 is only officially satisfied "
              f"by --device cuda on Colab; re-run there before treating this as a pass.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
