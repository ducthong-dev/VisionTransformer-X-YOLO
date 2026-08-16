"""Parameters, FLOPs, model size and latency (Reviewer #10.6).

Latency uses fixed warm-up and timed iteration counts on one device at one batch
size and resolution, and reports mean and standard deviation. The device string
is recorded so numbers collected on different hardware are never silently mixed.
"""

from __future__ import annotations

import itertools
import platform
import time

import torch


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable),
            "params_frozen": int(total - trainable)}


def model_size_mb(model: torch.nn.Module) -> float:
    b = sum(p.numel() * p.element_size() for p in model.parameters())
    b += sum(buf.numel() * buf.element_size() for buf in model.buffers())
    return b / (1024 ** 2)


def _origin_device(model: torch.nn.Module) -> torch.device | None:
    """The device the model's tensors currently live on. None if it has no tensors."""
    for t in itertools.chain(model.parameters(), model.buffers()):
        return t.device
    return None


def _strip_thop_buffers(model: torch.nn.Module) -> int:
    """Remove the `total_ops`/`total_params` buffers thop attaches while profiling.

    [MEASURED] thop 0.1.1 leaves 90 float64 buffers behind on the YOLOv8n-cls
    baseline after `profile()` returns. They are instrumentation, not part of the
    architecture, and removing them restores the exact pre-call state. Leaving
    them attached also makes the model unmovable to Apple MPS, which has no
    float64 support.
    """
    removed = 0
    for _, mod in model.named_modules():
        for key in ("total_ops", "total_params"):
            if key in mod._buffers:
                del mod._buffers[key]
                removed += 1
            elif hasattr(mod, key):
                try:
                    delattr(mod, key)
                    removed += 1
                except AttributeError:
                    pass
    return removed


def count_flops(model: torch.nn.Module, input_shape=(1, 3, 224, 224), device="cpu") -> dict:
    """MACs/FLOPs via thop, profiled on `device` (CPU by default, and valid from
    any device since FLOPs are hardware-independent).

    Placement- and state-neutral: the model is returned exactly as it arrived.

    `nn.Module.to()` mutates the module IN PLACE -- rebinding the local name does
    not give the caller an independent copy. Profiling on CPU therefore used to
    leave the caller's model stranded on CPU, so a subsequent CUDA forward pass
    raised "Input type (torch.cuda.FloatTensor) and weight type
    (torch.FloatTensor) should be the same". The incoming device is captured and
    restored in a `finally` block, after thop's temporary buffers are stripped.

    Returns NaNs rather than failing if thop cannot trace.
    """
    origin = _origin_device(model)
    try:
        from thop import profile

        model.to(device).eval()
        x = torch.randn(*input_shape, device=device)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        result = {"macs": float(macs), "gflops": float(2 * macs / 1e9)}
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        result = {"macs": float("nan"), "gflops": float("nan"), "flops_error": str(exc)}
    finally:
        removed = _strip_thop_buffers(model)
        if origin is not None:
            model.to(origin)
    result["thop_buffers_removed"] = removed
    return result


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    device: str = "cpu",
    batch_size: int = 1,
    img_size: int = 224,
    warmup: int = 10,
    iters: int = 50,
) -> dict:
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device.startswith("mps"):
            torch.mps.synchronize()

    for _ in range(warmup):
        model(x)
    sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)

    t = torch.tensor(times)
    mean = float(t.mean())
    return {
        "device": device,
        "batch_size": batch_size,
        "img_size": img_size,
        "warmup_iters": warmup,
        "timed_iters": iters,
        "latency_ms_mean": mean,
        "latency_ms_std": float(t.std()),
        "latency_ms_per_image": mean / batch_size,
        "throughput_img_per_s": (batch_size * 1000.0 / mean) if mean > 0 else float("nan"),
    }


def hardware_info(device: str) -> dict:
    info = {
        "device": device,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "torch": torch.__version__,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    info.update(timing_provenance(device))
    return info


def timing_provenance(device: str) -> dict:
    """Mark whether timings from this device may be quoted as deployment evidence.

    Latency, throughput and FPS are only reportable from the primary CUDA
    training environment. Numbers collected on a development laptop (CPU or Apple
    MPS) are for sanity-checking the implementation and must never reach the
    manuscript as performance evidence.
    """
    reportable = device.startswith("cuda")
    return {
        "timings_reportable": reportable,
        "timing_note": (
            "CUDA device: latency and throughput are reportable."
            if reportable
            else f"NON-CUDA device ({device}): DEVELOPMENT SANITY CHECK ONLY. "
                 "Do not quote these timings in the manuscript. Re-measure on the "
                 "primary CUDA environment."
        ),
    }


# --------------------------------------------------------------------------- #
# Device-placement verification.
#
# A benchmark that silently runs on the wrong device is worse than one that
# crashes: it either dies mid-run or reports timings for hardware that was never
# used. These helpers make placement an explicit, checked precondition of every
# measurement rather than an assumption.
# --------------------------------------------------------------------------- #


def canonical_device(device) -> torch.device:
    """Resolve a device the way tensor allocation does, so 'cuda' == 'cuda:0'.

    Without this, a model moved with `.to("cuda")` -- whose parameters then report
    `cuda:0` -- would compare unequal to the requested string and every check
    would fail spuriously. The same applies to 'mps' vs 'mps:0'. `cpu` and `meta`
    tensors always report a `None` index, so they are left alone.
    """
    d = torch.device(device)
    if d.index is not None or d.type in ("cpu", "meta"):
        return d
    if d.type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device(d.type, 0)


def model_tensor_devices(model: torch.nn.Module) -> dict:
    """Every device the model's tensors live on, recursively.

    `parameters()` and `buffers()` already recurse through registered submodules,
    which is exactly the set `.to()` moves -- so this sees inside wrappers such as
    Ultralytics' ClassificationModel, not just the outer custom front-end.

    Tensors held as plain Python attributes are reported separately under
    `unregistered_tensor_attrs`, because `.to()` does not move them and at least
    one is legitimately left behind: [MEASURED] `ClassificationModel.stride` is a
    bare `torch.Tensor` attribute that stays on CPU even in a perfectly healthy
    CUDA run, and takes no part in the classification forward pass. Treating it as
    a fatal mismatch would fail every candidate, so it is surfaced as a diagnostic
    instead of an error.
    """
    unregistered = []
    for name, mod in model.named_modules():
        registered = set(dict(mod.named_parameters(recurse=False))) | set(
            dict(mod.named_buffers(recurse=False)))
        for attr, value in vars(mod).items():
            if (isinstance(value, torch.Tensor) and not attr.startswith("_")
                    and attr not in registered):
                unregistered.append({"module": name or "<root>", "attr": attr,
                                     "device": str(value.device),
                                     "shape": list(value.shape)})
    return {
        "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
        "buffer_devices": sorted({str(b.device) for b in model.buffers()}),
        "n_parameters": sum(1 for _ in model.parameters()),
        "n_buffers": sum(1 for _ in model.buffers()),
        "unregistered_tensor_attrs": unregistered,
    }


def assert_model_on_device(model: torch.nn.Module, device, context: str = "") -> dict:
    """Fail clearly unless every parameter and buffer is on `device`.

    Returns the placement audit so callers can record it as evidence that the
    measurement really ran where it claims to have run.
    """
    target = canonical_device(device)
    offenders = [
        f"{name} on {t.device}"
        for name, t in itertools.chain(model.named_parameters(), model.named_buffers())
        if canonical_device(t.device) != target
    ]
    audit = model_tensor_devices(model)
    audit["target_device"] = str(target)
    audit["verified"] = not offenders
    if offenders:
        where = f" [{context}]" if context else ""
        more = f" (+{len(offenders) - 5} more)" if len(offenders) > 5 else ""
        raise RuntimeError(
            f"device placement check failed{where}: requested {target}, but "
            f"{len(offenders)} model tensor(s) are on another device. "
            f"First offenders: {', '.join(offenders[:5])}{more}. "
            "nn.Module.to() is in-place, so any step that profiles the model on a "
            "different device (e.g. count_flops on CPU) must be followed by "
            "model.to(device) before the next measurement."
        )
    return audit


HARDWARE_INDEPENDENT_FIELDS = (
    "params_total", "params_trainable", "params_frozen", "params_frontend",
    "model_size_mb", "macs", "gflops",
)


def full_profile(model, device="cpu", batch_size=1, img_size=224, in_channels=3) -> dict:
    out = count_parameters(model)
    out["model_size_mb"] = model_size_mb(model)
    out.update(count_flops(model, (1, in_channels, img_size, img_size), device="cpu"))
    out.update(measure_latency(model, device=device, batch_size=batch_size, img_size=img_size))
    out["hardware"] = hardware_info(device)
    return out


# --------------------------------------------------------------------------- #
# Peak-memory instrumentation for the training loop.
#
# Instrumentation only: these helpers read counters and never influence
# allocation, scheduling or numerics. They are no-ops on non-CUDA devices, so the
# trainer stays portable on CPU and Apple MPS.
#
# torch.mps has no equivalent of max_memory_allocated()/max_memory_reserved()
# with the same semantics, so MPS is deliberately reported as unavailable rather
# than approximated with a different quantity.
# --------------------------------------------------------------------------- #


def reset_peak_memory(device: str) -> None:
    """Reset CUDA peak-memory counters. Call immediately before training starts."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_stats(device: str) -> dict:
    """Peak CUDA memory since the last reset, in MiB.

    Returns `available: False` on any non-CUDA device -- never a fabricated
    number, and never a value borrowed from a different memory model.
    """
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return {"available": False, "device": str(device),
                "note": "peak memory is CUDA-only; not measured on this device"}
    return {
        "available": True,
        "device": str(device),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
    }


def current_peak_allocated_mb(device: str) -> float | None:
    """Running peak allocated MiB, for per-epoch logging. None when unavailable."""
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return None
    return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
