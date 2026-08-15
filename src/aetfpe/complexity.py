"""Parameters, FLOPs, model size and latency (Reviewer #10.6).

Latency uses fixed warm-up and timed iteration counts on one device at one batch
size and resolution, and reports mean and standard deviation. The device string
is recorded so numbers collected on different hardware are never silently mixed.
"""

from __future__ import annotations

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


def count_flops(model: torch.nn.Module, input_shape=(1, 3, 224, 224), device="cpu") -> dict:
    """MACs/FLOPs via thop. Returns NaNs rather than failing if thop cannot trace."""
    try:
        from thop import profile

        model = model.to(device).eval()
        x = torch.randn(*input_shape, device=device)
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return {"macs": float(macs), "gflops": float(2 * macs / 1e9)}
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"macs": float("nan"), "gflops": float("nan"), "flops_error": str(exc)}


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
