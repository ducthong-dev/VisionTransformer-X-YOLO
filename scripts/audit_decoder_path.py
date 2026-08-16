#!/usr/bin/env python
"""Layer-by-layer audit of the C2-* feature-space path (documentation only).

Traces every shape-changing operation between the input image and the tensor
handed to the classifier, for C2-7 / C2-14 / C2-28. Nothing is redesigned,
removed or optimised: this script only observes.

Implicit resizes are caught by patching the functional API (`F.interpolate`,
`F.adaptive_avg_pool2d`, `F.upsample`), so a resize that appears nowhere as a
registered module still shows up in the trace.

MAC counts are hardware-independent and computed analytically per layer:
    Conv2d           MACs = Cout * Hout * Wout * (Cin/groups) * kh * kw
    ConvTranspose2d  MACs = Cin  * Hin  * Win  * (Cout/groups) * kh * kw
Elementwise layers (BN, ReLU, Sigmoid) are reported by element count, not MACs.

    LOCAL   python scripts/audit_decoder_path.py

Shapes and MACs are hardware-independent, so this is valid from any device; it
runs on CPU with scratch weights and needs no dataset and no downloads.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.models import build_model  # noqa: E402

CANDIDATES = {"C2-7": None, "C2-14": 3, "C2-28": 2}

TRACE: list[dict] = []


def shape(t) -> str:
    return "x".join(str(int(v)) for v in t.shape) if torch.is_tensor(t) else "?"


def conv_macs(m, x, y) -> int:
    if isinstance(m, nn.ConvTranspose2d):
        return int(x.shape[1] * x.shape[2] * x.shape[3]
                   * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])
    if isinstance(m, nn.Conv2d):
        return int(y.shape[1] * y.shape[2] * y.shape[3]
                   * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])
    return 0


def describe_op(m) -> tuple[str, str, str, str]:
    """(op, kernel, stride, padding) as printable strings."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        k = "x".join(str(v) for v in m.kernel_size)
        s = "x".join(str(v) for v in m.stride)
        p = "x".join(str(v) for v in m.padding)
        return type(m).__name__, k, s, p
    return type(m).__name__, "-", "-", "-"


def hook(name: str):
    def fn(m, inputs, out):
        x = inputs[0]
        if not torch.is_tensor(x) or not torch.is_tensor(out):
            return
        op, k, s, p = describe_op(m)
        TRACE.append({
            "name": name, "op": op, "kernel": k, "stride": s, "padding": p,
            "in": shape(x), "out": shape(out),
            "macs": conv_macs(m, x, out),
            "elements": int(out.numel()) if op in ("BatchNorm2d", "ReLU", "Sigmoid") else 0,
        })
    return fn


def patch_functionals():
    """Log implicit resizes that are not registered modules."""
    orig_interp, orig_aap, orig_up = F.interpolate, F.adaptive_avg_pool2d, F.upsample

    def interp(x, *a, **kw):
        y = orig_interp(x, *a, **kw)
        TRACE.append({"name": "F.interpolate", "op": f"interpolate({kw.get('mode', 'nearest')})",
                      "kernel": "-", "stride": "-", "padding": "-",
                      "in": shape(x), "out": shape(y), "macs": 0,
                      "elements": int(y.numel()), "implicit": True})
        return y

    def aap(x, *a, **kw):
        y = orig_aap(x, *a, **kw)
        TRACE.append({"name": "F.adaptive_avg_pool2d", "op": "adaptive_avg_pool2d",
                      "kernel": "-", "stride": "-", "padding": "-",
                      "in": shape(x), "out": shape(y), "macs": 0,
                      "elements": int(x.numel()), "implicit": True})
        return y

    def up(x, *a, **kw):
        y = orig_up(x, *a, **kw)
        TRACE.append({"name": "F.upsample", "op": "upsample", "kernel": "-",
                      "stride": "-", "padding": "-", "in": shape(x), "out": shape(y),
                      "macs": 0, "elements": int(y.numel()), "implicit": True})
        return y

    F.interpolate, F.adaptive_avg_pool2d, F.upsample = interp, aap, up
    return orig_interp, orig_aap, orig_up


def restore(orig):
    F.interpolate, F.adaptive_avg_pool2d, F.upsample = orig


def trace_candidate(label: str, stage) -> dict:
    global TRACE
    cfg = dict(use_pe=True, use_tf=True, use_ae=True, fusion="linear",
               tf_backbone="mobilevit_xxs", ae_space="feature",
               num_classes=39, img_size=224, pretrained=False, vit_pretrained=False)
    if stage is not None:
        cfg["tf_stage"] = stage
    model = build_model(cfg).eval()

    # Shapes first, untraced, so the table below is exactly ONE forward pass.
    with torch.no_grad():
        out, parts = model.frontend(torch.rand(1, 3, 224, 224), return_parts=True)
        logits = model(torch.rand(1, 3, 224, 224))

    handles = []
    for name, mod in model.ae.named_modules():
        if not list(mod.children()):
            handles.append(mod.register_forward_hook(hook(f"ae.{name}")))
    for name, mod in model.tf.named_modules():
        if name in ("project", "norm"):
            handles.append(mod.register_forward_hook(hook(f"tf.{name}")))

    TRACE = []
    orig = patch_functionals()
    try:
        with torch.no_grad():
            model(torch.rand(1, 3, 224, 224))
    finally:
        restore(orig)
        for h in handles:
            h.remove()
    single = list(TRACE)

    return {
        "label": label,
        "grid": int(model.ae.grid),
        "ae_in_channels": int(model.ae_in_channels),
        "latent": list(parts["latent"].shape),
        "classifier_input": list(out.shape),
        "logits": list(logits.shape),
        "describe_decoder": model.ae.describe()["decoder"],
        "n_decoder_upsample_stages": len(model.ae.decoder) - 1,
        "trace": single,
        "ae_total_macs": sum(r["macs"] for r in single),
    }


def main() -> int:
    results = []
    for label, stage in CANDIDATES.items():
        r = trace_candidate(label, stage)
        results.append(r)

        print(f"\n{'=' * 108}\n{label}   grid {r['grid']}x{r['grid']}   "
              f"AE input channels {r['ae_in_channels']}\n{'=' * 108}")
        print(f"{'#':>3} {'layer':<26}{'operation':<22}{'k':>5}{'s':>4}{'p':>4}"
              f"{'input':>16}{'output':>16}{'MMACs':>10}")
        print("-" * 108)
        for i, row in enumerate(r["trace"], 1):
            mark = " *" if row.get("implicit") else "  "
            macs = f"{row['macs'] / 1e6:.3f}" if row["macs"] else "-"
            print(f"{i:>3}{mark}{row['name']:<24}{row['op']:<22}{row['kernel']:>5}"
                  f"{row['stride']:>4}{row['padding']:>4}{row['in']:>16}{row['out']:>16}{macs:>10}")
        print("-" * 108)
        print(f"  * = implicit resize (not a registered module)")
        print(f"  AE total: {r['ae_total_macs'] / 1e6:.3f} MMACs "
              f"= {2 * r['ae_total_macs'] / 1e9:.4f} GFLOPs")
        print(f"  latent {r['latent']}  ->  classifier_input {r['classifier_input']}"
              f"  ->  logits {r['logits']}")
        print(f"  describe() decoder string : {r['describe_decoder']!r}")
        print(f"  actual upsampling stages  : {r['n_decoder_upsample_stages']}")

    print(f"\n{'=' * 108}\nSUMMARY\n{'=' * 108}")
    print(f"{'candidate':<10}{'grid':>6}{'AE in':>7}{'upsample stages':>17}"
          f"{'AE GFLOPs':>12}{'implicit resizes':>18}")
    for r in results:
        n_imp = sum(1 for x in r["trace"] if x.get("implicit"))
        print(f"{r['label']:<10}{r['grid']:>6}{r['ae_in_channels']:>7}"
              f"{r['n_decoder_upsample_stages']:>17}"
              f"{2 * r['ae_total_macs'] / 1e9:>12.4f}{n_imp:>18}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
