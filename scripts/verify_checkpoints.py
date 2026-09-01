#!/usr/bin/env python
"""Verify downloaded campaign checkpoints before any of them is evaluated.

Acquisition and verification are deliberately separate: this script assumes the
artifacts are already mirrored under `results/campaign/scientific/<ID>/` and decides,
independently of how they got there, whether each one may be evaluated.

    python scripts/verify_checkpoints.py
    python scripts/verify_checkpoints.py --root results/campaign/scientific --strict

A checkpoint is REFUSED unless all of the following hold:

  * provenance says namespace=scientific, smoke_test=false, full_data=true
  * dataset_sha256 matches the campaign's training dataset
  * epochs_requested == 50 and the run actually completed 50 epochs
  * the loaded tensor dict is the `best_val_top1` selection -- `val_top1` and `epoch`
    inside the file agree with `train_summary.json`'s recorded best, rather than being
    trusted because the file happens to be named `checkpoint.pt`
  * the class list is the expected 39 and matches the evaluation dataset's ordering

Nothing here reads a test split, and no artifact is modified. Refusals are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.provenance import compare, refuse  # noqa: E402

EXPECTED = {
    "namespace": "scientific",
    "smoke_test": False,
    "full_data": True,
    "epochs_requested": 50,
    "dataset_sha256": "4f8a8332c3900e318f172c633a2aa5ec8b475174b76152a9db828173bae1897d",
}
EXPECTED_EPOCHS = 50
EXPECTED_CLASSES = 39
VAL_TOL = 1e-9

# Physical runs only. F3/F5/F5_clean are aliases and are never evaluated separately.
ALIASES = {"F3": "A3", "F5": "A5", "F5_clean": "D1"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_one(run_dir: str, classes_ref: list[str] | None) -> dict:
    rid = os.path.basename(run_dir.rstrip("/"))
    out: dict = {"run_id": rid, "dir": run_dir, "refusals": [], "warnings": []}

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    prov_path = os.path.join(run_dir, "run_provenance.json")
    summ_path = os.path.join(run_dir, "train_summary.json")

    for label, p in (("checkpoint.pt", ckpt_path), ("run_provenance.json", prov_path),
                     ("train_summary.json", summ_path)):
        if not os.path.exists(p):
            out["refusals"].append(f"missing {label}")
    if out["refusals"]:
        out["verdict"] = "REFUSED"
        return out

    prov = json.load(open(prov_path))
    summ = json.load(open(summ_path))

    # ---- provenance identity ---------------------------------------------- #
    mism = compare(EXPECTED, prov, fields=tuple(EXPECTED))
    out["refusals"] += mism
    out["provenance"] = {k: prov.get(k) for k in
                         ("run_id", "namespace", "smoke_test", "full_data",
                          "epochs_requested", "timing_basis", "git_commit",
                          "config_sha256", "protocol_sha256", "dataset_sha256")}
    if prov.get("run_id") != rid:
        out["refusals"].append(f"provenance run_id {prov.get('run_id')!r} != directory {rid!r}")

    # ---- completion -------------------------------------------------------- #
    status = summ.get("status")
    done = summ.get("epochs_completed")
    out["status"] = status
    out["epochs_completed"] = done
    if status != "completed":
        out["refusals"].append(f"train_summary status is {status!r}, not 'completed'")
    if done != EXPECTED_EPOCHS:
        out["refusals"].append(f"epochs_completed={done}, expected {EXPECTED_EPOCHS}")

    # ---- the artifact really is the best-val selection ---------------------- #
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("model", "cfg", "classes", "epoch", "val_top1"):
        if k not in ck:
            out["refusals"].append(f"checkpoint missing key {k!r}")
    if out["refusals"]:
        out["verdict"] = "REFUSED"
        return out

    best_recorded = summ.get("best_val_top1")
    out["selected_epoch"] = int(ck["epoch"])
    out["selected_val_top1"] = float(ck["val_top1"])
    out["recorded_best_val_top1"] = best_recorded
    if best_recorded is None:
        out["refusals"].append("train_summary has no best_val_top1 to check the selection against")
    elif abs(float(ck["val_top1"]) - float(best_recorded)) > VAL_TOL:
        out["refusals"].append(
            f"checkpoint val_top1={ck['val_top1']!r} != recorded best {best_recorded!r} "
            "-- this file is NOT the best-val selection")
    if not (1 <= int(ck["epoch"]) <= EXPECTED_EPOCHS):
        out["refusals"].append(f"selected epoch {ck['epoch']} outside 1..{EXPECTED_EPOCHS}")

    # ---- classes ----------------------------------------------------------- #
    cls = list(ck["classes"])
    out["num_classes"] = len(cls)
    if len(cls) != EXPECTED_CLASSES:
        out["refusals"].append(f"{len(cls)} classes, expected {EXPECTED_CLASSES}")
    if classes_ref is not None and cls != classes_ref:
        out["refusals"].append("class ordering differs from the evaluation dataset")

    # ---- integrity ---------------------------------------------------------- #
    out["checkpoint_path"] = ckpt_path
    out["checkpoint_sha256"] = sha256_file(ckpt_path)
    out["checkpoint_bytes"] = os.path.getsize(ckpt_path)
    out["num_tensors"] = len(ck["model"])
    out["total_params"] = int(sum(v.numel() for v in ck["model"].values() if hasattr(v, "numel")))

    if prov.get("git_commit") and summ.get("environment", {}).get("git_dirty"):
        out["warnings"].append("trained from a dirty working tree (commit recorded, tree not exact)")

    out["verdict"] = "REFUSED" if out["refusals"] else "ACCEPTED"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/campaign/scientific")
    ap.add_argument("--out", default="results/campaign/checkpoint_verification.json")
    ap.add_argument("--data-root", default=None,
                    help="dataset root; class ordering is cross-checked against test/")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any run is refused")
    args = ap.parse_args()

    classes_ref = None
    if args.data_root:
        t = os.path.join(args.data_root, "test")
        if os.path.isdir(t):
            classes_ref = sorted(d for d in os.listdir(t) if os.path.isdir(os.path.join(t, d)))

    if not os.path.isdir(args.root):
        print(f"ERROR: {args.root} does not exist -- no checkpoints have been acquired yet.",
              file=sys.stderr)
        print("Nothing verified, nothing assumed.", file=sys.stderr)
        return 2

    runs = sorted(d for d in os.listdir(args.root)
                  if os.path.isdir(os.path.join(args.root, d)) and not d.startswith("."))
    results = []
    for rid in runs:
        if rid in ALIASES:
            print(f"{rid:10s} SKIPPED (alias of {ALIASES[rid]}; never evaluated separately)")
            continue
        r = verify_one(os.path.join(args.root, rid), classes_ref)
        results.append(r)
        mark = "OK " if r["verdict"] == "ACCEPTED" else "XX "
        extra = ""
        if r["verdict"] == "ACCEPTED":
            extra = (f"ep {r['selected_epoch']:>2}/{EXPECTED_EPOCHS}  "
                     f"val_top1={r['selected_val_top1']:.7f}  "
                     f"{r['total_params']:,} params  sha={r['checkpoint_sha256'][:12]}…")
        print(f"{mark}{rid:10s} {r['verdict']:8s} {extra}")
        for m in r["refusals"]:
            print(f"      REFUSED: {m}")
        for w in r["warnings"]:
            print(f"      note: {w}")

    accepted = [r for r in results if r["verdict"] == "ACCEPTED"]
    refused = [r for r in results if r["verdict"] != "ACCEPTED"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"expected": EXPECTED, "expected_epochs": EXPECTED_EPOCHS,
                   "aliases": ALIASES, "accepted": [r["run_id"] for r in accepted],
                   "refused": [r["run_id"] for r in refused], "runs": results}, fh, indent=2)

    print(f"\n{len(accepted)} accepted, {len(refused)} refused -> {args.out}")
    if refused:
        print("Refused runs are reported, not silently skipped:", ", ".join(r["run_id"] for r in refused))
    return 1 if (args.strict and refused) else 0


if __name__ == "__main__":
    raise SystemExit(main())
