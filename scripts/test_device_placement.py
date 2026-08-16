#!/usr/bin/env python
"""Regression test for the benchmark harness's device placement.

Guards the defect that killed the official T4 benchmark run: `count_flops()`
profiles on CPU, `nn.Module.to()` mutates in place, and the model was left on CPU
while the benchmark went on to feed it CUDA tensors -- so the first parameterised
layer raised

    RuntimeError: Input type (torch.cuda.FloatTensor) and weight type
                  (torch.FloatTensor) should be the same

Runs entirely on CPU with scratch weights: no dataset, no downloads, no training,
no CUDA required. The mismatch-detection tests use the `meta` device, which exists
on every platform, so they are real assertions rather than skipped ones.

    LOCAL / COLAB   python scripts/test_device_placement.py

CUDA-specific behaviour is NOT asserted here -- it cannot be tested honestly on a
machine without CUDA. What is tested is device-agnostic: that a model profiled on
one device is restored to the other, and that a mismatch is detected and reported.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.complexity import (  # noqa: E402
    assert_model_on_device, canonical_device, count_flops, model_tensor_devices,
)
from aetfpe.models import build_model  # noqa: E402

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def build(**over):
    """Scratch weights only: placement is independent of weight values, and this
    keeps the test offline and fast."""
    cfg = dict(use_pe=False, use_tf=False, use_ae=False, fusion="identity",
               num_classes=39, img_size=224, pretrained=False, vit_pretrained=False)
    cfg.update(over)
    return build_model(cfg).eval()


def alt_device() -> str | None:
    """A second *real* device, if this machine has one. None on CPU-only hosts."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return None


# --------------------------------------------------------------------------- #

def test_cpu_placement() -> None:
    """--device cpu puts every parameter and buffer on CPU."""
    print("\n[1] every parameter/buffer on CPU after .to('cpu')")
    for label, over in (("BASELINE", {}),
                        ("C2-28", dict(use_pe=True, use_tf=True, use_ae=True,
                                       fusion="linear", tf_backbone="mobilevit_xxs",
                                       ae_space="feature", tf_stage=2))):
        model = build(**over).to("cpu")
        audit = assert_model_on_device(model, "cpu", f"{label} cpu")
        check(f"{label}: parameters all on cpu", audit["parameter_devices"] == ["cpu"],
              str(audit["parameter_devices"]))
        check(f"{label}: buffers all on cpu",
              audit["buffer_devices"] in ([], ["cpu"]), str(audit["buffer_devices"]))
        check(f"{label}: audit verified flag set", audit["verified"] is True)
        check(f"{label}: forward runs on cpu",
              tuple(model(torch.rand(2, 3, 224, 224)).shape) == (2, 39))


def test_count_flops_is_state_neutral() -> None:
    """count_flops must return the model exactly as it arrived.

    thop attaches float64 `total_ops`/`total_params` buffers while profiling; if
    they are left behind they pollute the model and, on Apple MPS, make it
    unmovable (no float64 support).
    """
    print("\n[2a] count_flops leaves the model's tensor set unchanged")
    model = build().to("cpu")
    before = set(dict(model.named_buffers()))
    res = count_flops(model, (1, 3, 224, 224), device="cpu")
    after = set(dict(model.named_buffers()))
    check("no thop buffers left attached", before == after,
          f"leaked {sorted(after - before)[:3]}")
    check("thop buffer cleanup is reported", res.get("thop_buffers_removed", 0) > 0,
          f"removed={res.get('thop_buffers_removed')}")
    check("no float64 tensors remain",
          not [n for n, b in model.named_buffers() if b.dtype == torch.float64])
    check("FLOPs still measured", res["gflops"] > 0, f"{res['gflops']:.4f} GFLOPs")


def test_count_flops_is_placement_neutral() -> None:
    """THE REGRESSION: profiling FLOPs on CPU must not strand the model there."""
    print("\n[2b] count_flops(device='cpu') restores the model's original device")
    dev = alt_device()
    if dev is None:
        model = build().to("cpu")
        count_flops(model, (1, 3, 224, 224), device="cpu")
        audit = model_tensor_devices(model)
        check("cpu-only host: model still on cpu after count_flops",
              audit["parameter_devices"] == ["cpu"])
        print("  NOTE  no second device on this host, so the cross-device restore "
              "could not be exercised. Not fabricated as a pass.")
        return

    model = build().to(dev)
    before = model_tensor_devices(model)["parameter_devices"]
    count_flops(model, (1, 3, 224, 224), device="cpu")
    after = model_tensor_devices(model)["parameter_devices"]
    check(f"model restored to {dev} after CPU FLOPs profiling", before == after,
          f"before={before} after={after}")
    try:
        model(torch.rand(2, 3, 224, 224, device=dev))
        check(f"forward on {dev} still runs after count_flops", True)
    except RuntimeError as exc:
        check(f"forward on {dev} still runs after count_flops", False, str(exc)[:90])


def strand(module, device: str = "meta", limit: int | None = None) -> list[str]:
    """Move parameters of `module` to another device, simulating a partial .to().

    Rebinds the entries in `_parameters` rather than assigning to `.data`, which
    refuses a cross-device-type swap. `meta` is used because it exists on every
    platform, so the mismatch test is real on a CPU-only host too.
    """
    moved = []
    for qual, sub in module.named_modules():
        for pname, p in list(sub._parameters.items()):
            if p is None:
                continue
            sub._parameters[pname] = torch.nn.Parameter(
                p.detach().to(device), requires_grad=p.requires_grad)
            moved.append(f"{qual}.{pname}" if qual else pname)
            if limit is not None and len(moved) >= limit:
                return moved
    return moved


def test_mismatch_is_detected() -> None:
    """A model that is not wholly on the target device must fail loudly."""
    print("\n[3] mismatched placement is detected and named")

    # One stranded parameter, deep inside the Ultralytics wrapper.
    model = build().to("cpu")
    name = "classifier." + strand(model.classifier, limit=1)[0]
    try:
        assert_model_on_device(model, "cpu", "deliberate mismatch")
        check("stranded classifier parameter raises", False, "no exception raised")
    except RuntimeError as exc:
        check("stranded classifier parameter raises", True)
        check("error names the offending tensor", name in str(exc), str(exc)[:90])
        check("error names the requested device", "cpu" in str(exc))

    # The exact shape of the original bug: the custom front-end is on the target
    # device but the wrapped classifier is not. Moving only the outer module is
    # what a naive fix does, so this is the case that must not pass silently.
    model = build(use_pe=True, use_tf=True, use_ae=True, fusion="linear",
                  tf_backbone="mobilevit_xxs", ae_space="feature", tf_stage=2).to("cpu")
    strand(model.classifier)
    try:
        assert_model_on_device(model, "cpu", "frontend-only move")
        check("front-end moved but classifier stranded raises", False,
              "no exception raised")
    except RuntimeError as exc:
        check("front-end moved but classifier stranded raises", True)
        check("error points at classifier tensors", "classifier" in str(exc),
              str(exc)[:90])


def test_canonical_device() -> None:
    """'cuda' and 'cuda:0' must not be reported as a mismatch."""
    print("\n[4] device canonicalisation")
    check("cpu resolves to cpu", canonical_device("cpu") == torch.device("cpu"))
    check("torch.device passthrough",
          canonical_device(torch.device("cpu")) == torch.device("cpu"))
    check("meta keeps its None index", canonical_device("meta") == torch.device("meta"))
    dev = alt_device()
    if dev:
        # The bare name must match what a tensor moved with .to(dev) reports.
        probe = torch.zeros(1, device=dev).device
        check(f"'{dev}' matches the device a tensor reports ({probe})",
              canonical_device(dev) == canonical_device(probe))
    if torch.cuda.is_available():
        check("cuda resolves to an indexed device",
              canonical_device("cuda") == canonical_device("cuda:0"))
    else:
        # Without CUDA the indexed form cannot be resolved against a live device;
        # assert only the no-op path rather than inventing a CUDA result.
        check("cuda:0 passthrough without CUDA present",
              canonical_device("cuda:0") == torch.device("cuda", 0))


def test_unregistered_attrs_reported_not_fatal() -> None:
    """Ultralytics keeps `stride` as a bare attribute .to() never moves."""
    print("\n[5] unregistered tensor attributes are reported, not fatal")
    model = build().to("cpu")
    audit = assert_model_on_device(model, "cpu")
    attrs = {(a["module"], a["attr"]) for a in audit["unregistered_tensor_attrs"]}
    check("ClassificationModel.stride surfaced in the audit",
          ("classifier", "stride") in attrs, str(sorted(attrs)))
    check("its presence does not fail the check", audit["verified"] is True)


def test_benchmark_call_order() -> None:
    """The harness's real sequence must survive end to end."""
    print("\n[6] benchmark call order: build -> to -> count_flops -> assert -> forward")
    dev = alt_device() or "cpu"
    model = build().to(dev).eval()
    assert_model_on_device(model, dev, "after build")
    count_flops(model, (1, 3, 224, 224), device="cpu")
    model.to(dev).eval()
    audit = assert_model_on_device(model, dev, "before trace_shapes")
    ok = True
    try:
        model(torch.rand(2, 3, 224, 224, device=dev))
    except RuntimeError as exc:
        ok = False
        print("   ", str(exc)[:90])
    check(f"full harness order runs on {dev}", ok and audit["verified"])


def main() -> int:
    print(f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | "
          f"second device={alt_device() or 'none (CPU-only host)'}")
    test_cpu_placement()
    test_count_flops_is_state_neutral()
    test_count_flops_is_placement_neutral()
    test_mismatch_is_detected()
    test_canonical_device()
    test_unregistered_attrs_reported_not_fatal()
    test_benchmark_call_order()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
