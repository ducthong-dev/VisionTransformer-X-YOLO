"""Config loading, provenance capture, and the frozen training protocol.

One protocol object is shared by every arm. If a field is not in `Protocol`, it
is not allowed to vary between arms -- that is the mechanism by which the
"identical experimental conditions" claim (Reviewer #10.7) is made checkable
rather than asserted.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Protocol:
    """Frozen across all arms. Values follow the recovered baseline log.

    log-org-280223 resolved `optimizer=auto` to AdamW(lr=7.14e-4, momentum=0.9)
    with decay groups 0.0 / 0.001 / 0.0, seed 0, deterministic, batch 128,
    imgsz 224, 30 epochs. Those are reproduced here rather than the values the
    manuscript claims (lr 0.01, momentum 0.937), which the log explicitly says
    were ignored.
    """

    img_size: int = 224
    batch_size: int = 128
    epochs: int = 30
    optimizer: str = "adamw"
    lr: float = 0.000714
    weight_decay: float = 0.001
    betas: tuple = (0.9, 0.999)
    warmup_epochs: float = 3.0
    label_smoothing: float = 0.0
    seed: int = 0
    deterministic: bool = True
    checkpoint_selection: str = "best_val_top1"
    num_workers: int = 4
    amp: bool = False
    # auto-encoder stage
    ae_beta: float = 0.001          # KL sparsity weight
    ae_rho: float = 0.05            # sparsity target
    ae_loss_weight: float = 10.0    # weight of the AE loss in the joint objective
    ae_warmup_epochs: int = 3       # reconstruction-only epochs before joint training


# --------------------------------------------------------------------------- #
# Environment roots.
#
# Committed configs never contain absolute paths. They reference ${DATA_ROOT},
# ${OUTPUT_ROOT}, ${CACHE_ROOT} and ${MODEL_ROOT}, which are resolved here, so
# the same config file runs unchanged on a laptop and on Colab.
#
#   LOCAL   export DATA_ROOT="$HOME/path/to/Plant_leaf_diseases_dataset"
#   COLAB   export DATA_ROOT=/content/data/Plant_leaf_diseases_dataset
#
# A git-ignored configs/local.yaml may also set them, for local development.
# --------------------------------------------------------------------------- #

ENV_DEFAULTS = {
    "DATA_ROOT": "data/dataset",
    "OUTPUT_ROOT": "results",
    "CACHE_ROOT": ".cache",
    "MODEL_ROOT": "weights",
}


def _local_overrides() -> dict:
    path = os.path.join(REPO_ROOT, "configs", "local.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return (yaml.safe_load(fh) or {}).get("env", {}) or {}


def resolve_roots() -> dict:
    """Effective values for the four roots: env var > configs/local.yaml > default."""
    local = _local_overrides()
    return {k: os.environ.get(k) or local.get(k) or v for k, v in ENV_DEFAULTS.items()}


def expand_paths(obj):
    """Recursively expand ${VAR} and ~ in every string of a loaded config."""
    roots = resolve_roots()
    if isinstance(obj, dict):
        return {k: expand_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_paths(v) for v in obj]
    if isinstance(obj, str):
        out = obj
        for k, v in roots.items():
            out = out.replace(f"${{{k}}}", v).replace(f"${k}", v)
        return os.path.expanduser(os.path.expandvars(out))
    return obj


def load_yaml(path: str) -> dict:
    with open(path) as fh:
        return expand_paths(yaml.safe_load(fh) or {})


def load_experiment(path: str) -> dict:
    """Load an experiment config, applying `extends` (one level) if present."""
    cfg = load_yaml(path)
    parent = cfg.pop("extends", None)
    if parent:
        base_path = parent if os.path.isabs(parent) else os.path.join(os.path.dirname(path), parent)
        base = load_experiment(base_path)
        merged = {**base, **cfg}
        for key in ("model", "protocol", "data"):
            if isinstance(base.get(key), dict) and isinstance(cfg.get(key), dict):
                merged[key] = {**base[key], **cfg[key]}
        cfg = merged
    return cfg


def build_protocol(cfg: dict) -> Protocol:
    return Protocol(**(cfg.get("protocol") or {}))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(out)
    except Exception:  # noqa: BLE001
        return False


def environment_info() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
    }


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_run_provenance(out_dir: str, cfg: dict, protocol: Protocol, extra: dict | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    payload = {"environment": environment_info(), "protocol": asdict(protocol)}
    if extra:
        payload.update(extra)
    with open(os.path.join(out_dir, "environment.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
