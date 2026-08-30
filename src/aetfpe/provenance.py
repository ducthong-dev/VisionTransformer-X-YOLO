"""Run identity: what makes two runs the same experiment, and what refuses them.

A checkpoint on Drive is only safe to adopt or resume from if it was produced by
*this* experiment. "This experiment" is not the run ID alone -- a 4-epoch,
4-images-per-class smoke run and the 30-epoch full-data scientific run share the
ID `A0` and write the same file names. Adopting one for the other silently
publishes a plumbing test as a result.

So identity is a record, not a name:

    run ID | namespace | smoke_test | epochs requested | per-class limits
           | full-data status | config hash | protocol hash | dataset hash

`build()` stamps that record into every run. `compare()` refuses on any
mismatch, loudly, rather than falling back to "start from epoch 0" or
"COMPLETED". Nothing here touches training semantics: it only decides whether an
artifact may be reused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time

PROVENANCE_VERSION = 1
PROVENANCE_FILE = "run_provenance.json"

NS_PREFLIGHT = "preflight"
NS_SCIENTIFIC = "scientific"
NAMESPACES = (NS_PREFLIGHT, NS_SCIENTIFIC)

# Fields that must be identical for one run to adopt or resume another.
IDENTITY_FIELDS = (
    "run_id",
    "namespace",
    "smoke_test",
    "epochs_requested",
    "limit_per_class",
    "limit_train_per_class",
    "limit_val_per_class",
    "full_data",
    "config_sha256",
    "protocol_sha256",
    "dataset_sha256",
)

# Config keys that legitimately differ between machines and therefore must not
# enter the config hash. The dataset is identified by its own fingerprint, and
# num_classes is derived from that dataset rather than authored in the YAML.
_VOLATILE_CONFIG_KEYS = (("data", "root"), ("model", "num_classes"))


class ProvenanceMismatch(Exception):
    """Raised when an artifact does not belong to the run asking for it."""


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_of(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def short(h) -> str:
    return (h or "?")[:12]


def config_identity(cfg: dict) -> str:
    """Hash of the fully resolved experiment config, machine-independent.

    Overrides are already merged into `cfg` by the time this runs, so an
    `--override protocol.ae_loss_weight=1` run cannot masquerade as the frozen
    one: it hashes differently.
    """
    c = copy.deepcopy(cfg)
    for parent, key in _VOLATILE_CONFIG_KEYS:
        if isinstance(c.get(parent), dict):
            c[parent].pop(key, None)
    return sha256_of(c)


def protocol_identity(protocol) -> str:
    """Hash of the frozen protocol as actually applied (post `--epochs` override)."""
    from dataclasses import asdict, is_dataclass

    return sha256_of(asdict(protocol) if is_dataclass(protocol) else dict(protocol))


def dataset_identity(train_fp: dict, val_fp: dict) -> dict:
    """Content identity of the split pair, independent of where it is mounted."""
    payload = {
        "train_listing_sha256": (train_fp or {}).get("listing_sha256"),
        "val_listing_sha256": (val_fp or {}).get("listing_sha256"),
        "train_images": (train_fp or {}).get("num_images"),
        "val_images": (val_fp or {}).get("num_images"),
        "num_classes": (train_fp or {}).get("num_classes"),
    }
    return {**payload, "sha256": sha256_of(payload)}


def build(*, run_id: str, namespace: str, smoke_test: bool, config_path: str,
          cfg: dict, protocol, epochs_requested: int,
          limit_per_class=None, limit_train_per_class=None, limit_val_per_class=None,
          train_fp: dict | None = None, val_fp: dict | None = None,
          train_images_used=None, val_images_used=None,
          git_commit: str = "", extra: dict | None = None) -> dict:
    if namespace not in NAMESPACES:
        raise ValueError(f"namespace must be one of {NAMESPACES}, got {namespace!r}")
    limits = (limit_per_class, limit_train_per_class, limit_val_per_class)
    ds = dataset_identity(train_fp or {}, val_fp or {})
    rec = {
        "provenance_version": PROVENANCE_VERSION,
        "run_id": run_id,
        "namespace": namespace,
        "smoke_test": bool(smoke_test),
        "config_path": config_path,
        "config_sha256": config_identity(cfg),
        "protocol_sha256": protocol_identity(protocol),
        "epochs_requested": int(epochs_requested),
        "limit_per_class": limit_per_class,
        "limit_train_per_class": limit_train_per_class,
        "limit_val_per_class": limit_val_per_class,
        "full_data": all(v is None for v in limits),
        "dataset": ds,
        "dataset_sha256": ds["sha256"],
        "train_images_used": train_images_used,
        "val_images_used": val_images_used,
        # Wall-clock measured under a per-class limit describes a different
        # workload than the scientific run. Carry that verdict with the run so a
        # cost model cannot pick the number up without also seeing the label.
        "timing_basis": "SMOKE_TIMING_ONLY" if (smoke_test or not all(v is None for v in limits))
                        else "FULL_DATA",
        "git_commit": git_commit,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        rec.update(extra)
    return rec


def compare(expected: dict, found: dict, fields=IDENTITY_FIELDS) -> list[str]:
    """Human-readable mismatches. Empty list == the artifact belongs to this run."""
    out = []
    if not found:
        return ["no provenance record found (artifact predates provenance stamping)"]
    for f in fields:
        if f not in expected:
            continue                      # caller deliberately does not constrain it
        e, g = expected.get(f), found.get(f)
        if f not in found:
            out.append(f"{f}: MISSING in artifact, expected {e!r}")
        elif e != g:
            out.append(f"{f}: artifact has {g!r}, this run requires {e!r}")
    return out


def refuse(kind: str, run_id: str, path: str, mismatches: list[str]) -> str:
    """The loud refusal text. Returned so callers can both print and raise it."""
    lines = [
        "",
        "=" * 78,
        f"REFUSED: {kind} for run {run_id}",
        "=" * 78,
        f"artifact: {path}",
        "provenance mismatch -- this artifact was produced by a DIFFERENT experiment:",
    ]
    lines += [f"  * {m}" for m in mismatches]
    lines += [
        "",
        "Nothing was adopted, resumed, or overwritten. Reusing it would publish one",
        "experiment's numbers under another experiment's name.",
        "=" * 78,
        "",
    ]
    return "\n".join(lines)


def load(path: str) -> dict:
    """Read a provenance record from a run directory or a file path."""
    if os.path.isdir(path):
        path = os.path.join(path, PROVENANCE_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 - an unreadable stamp is a refusal, not a crash
        return {}


def save(out_dir: str, rec: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, PROVENANCE_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=2, default=str)
    os.replace(tmp, path)
    return path


def is_smoke(rec: dict) -> bool:
    """True if the record describes anything other than a full-data run."""
    if not rec:
        return False
    return bool(rec.get("smoke_test")) or not rec.get("full_data", True) \
        or rec.get("namespace") == NS_PREFLIGHT


def describe(rec: dict) -> str:
    if not rec:
        return "(no provenance)"
    return (f"{rec.get('run_id')} ns={rec.get('namespace')} smoke={rec.get('smoke_test')} "
            f"epochs={rec.get('epochs_requested')} limit={rec.get('limit_per_class')} "
            f"full_data={rec.get('full_data')} cfg={short(rec.get('config_sha256'))} "
            f"proto={short(rec.get('protocol_sha256'))} data={short(rec.get('dataset_sha256'))}")
