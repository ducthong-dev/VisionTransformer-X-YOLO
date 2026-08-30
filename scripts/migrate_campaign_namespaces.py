#!/usr/bin/env python
"""Move a flat pre-namespace campaign tree into preflight/ and scientific/.

    python scripts/migrate_campaign_namespaces.py --drive-root <ROOT>            # dry run
    python scripts/migrate_campaign_namespaces.py --drive-root <ROOT> --apply

The old layout wrote every run to the same place regardless of what it was:

    <ROOT>/checkpoints/<RUN_ID>      <ROOT>/logs/<RUN_ID>.log
    <ROOT>/campaign/campaign_manifest.json

so the T4 architecture smoke tests (4 epochs, 4 images/class) landed in the same
directories the 30-epoch full-data runs will use, and the manifest reported them
COMPLETED. This script separates them by evidence rather than by name.

Nothing is deleted. Each run directory is COPIED into its namespace, and the
whole legacy tree is then MOVED to `<ROOT>/_legacy_pre_namespace/` so it stays
readable but can no longer be picked up by the campaign engine.

Classification is evidential, from what the run itself recorded:

    full data   metrics.csv train_n == environment.json train_fingerprint.num_images
    full epochs train_summary.json epochs_planned == the config's protocol epochs

Anything that is neither provably full nor provably partial goes to
`quarantine/` rather than to either namespace.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from aetfpe import provenance as prov  # noqa: E402

LEGACY_SUBDIRS = ("checkpoints", "logs", "campaign", "completed", "failed", "configs")


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _protocol_epochs(config_rel: str) -> int | None:
    try:
        from aetfpe.config import build_protocol, load_experiment
        return build_protocol(load_experiment(os.path.join(REPO_ROOT, config_rel))).epochs
    except Exception:  # noqa: BLE001
        return None


def classify(run_dir: str) -> dict:
    """Decide preflight / scientific / quarantine from the run's own record."""
    ev: dict = {"run_dir": run_dir}
    rec = prov.load(run_dir)
    if rec:
        ev["source"] = "run_provenance.json"
        ev["namespace"] = rec.get("namespace")
        ev["smoke_test"] = bool(rec.get("smoke_test"))
        ev["full_data"] = bool(rec.get("full_data"))
        ev["verdict"] = prov.NS_PREFLIGHT if prov.is_smoke(rec) else prov.NS_SCIENTIFIC
        return ev

    summary = _read_json(os.path.join(run_dir, "train_summary.json"))
    env = _read_json(os.path.join(run_dir, "environment.json"))
    ev["source"] = "train_summary.json + environment.json + metrics.csv"
    ev["status"] = summary.get("status")
    ev["epochs_planned"] = summary.get("epochs_planned")
    ev["epochs_completed"] = summary.get("epochs_completed")
    ev["config"] = summary.get("config")

    split_n = ((env.get("train_fingerprint") or {}).get("num_images"))
    ev["split_images"] = split_n

    trained_n = summary.get("train_images")
    if trained_n is None:
        mpath = os.path.join(run_dir, "metrics.csv")
        if os.path.exists(mpath):
            rows = list(csv.DictReader(open(mpath)))
            vals = [int(float(r["train_n"])) for r in rows if r.get("train_n")]
            trained_n = max(vals) if vals else None
    ev["trained_images"] = trained_n

    proto_ep = _protocol_epochs(ev["config"]) if ev.get("config") else None
    ev["protocol_epochs"] = proto_ep

    full_data = (split_n is not None and trained_n is not None and trained_n == split_n)
    full_epochs = (proto_ep is not None and ev.get("epochs_planned") == proto_ep)
    ev["full_data"] = full_data
    ev["full_epochs"] = full_epochs

    if split_n is None or trained_n is None:
        ev["verdict"] = "quarantine"
        ev["why"] = "cannot prove how many images this run trained on"
    elif not full_data:
        # A subset run is a smoke run whatever its epoch count, so this verdict
        # does not depend on being able to read the config.
        ev["verdict"] = prov.NS_PREFLIGHT
        why = f"trained on {trained_n:,} of {split_n:,} images"
        if proto_ep is not None:
            why += f", {ev.get('epochs_planned')} of {proto_ep} epochs"
        ev["why"] = why
    elif proto_ep is None:
        # Full data, but the config's epoch count could not be read -- so we
        # cannot tell a completed run from a truncated one. Quarantine rather
        # than guess in either direction.
        ev["verdict"] = "quarantine"
        ev["why"] = (f"trained on the full split ({trained_n:,}) but {ev.get('config')!r} "
                     "could not be loaded to check the epoch budget")
    elif full_epochs:
        ev["verdict"] = prov.NS_SCIENTIFIC
        ev["why"] = f"trained on the full split ({trained_n:,}) for the full {proto_ep} epochs"
    else:
        ev["verdict"] = prov.NS_PREFLIGHT
        ev["why"] = (f"trained on the full split ({trained_n:,}) but only "
                     f"{ev.get('epochs_planned')} of {proto_ep} epochs")
    return ev


def stamp_preflight(run_dir: str, ev: dict) -> None:
    """Mark migrated smoke evidence so it can never be adopted as science.

    The stamp deliberately omits the config, protocol and dataset hashes: with
    them missing, `provenance.compare()` refuses every adoption and resume
    attempt instead of having to reason about whether the values match.
    """
    prov.save(run_dir, {
        "provenance_version": prov.PROVENANCE_VERSION,
        "run_id": os.path.basename(run_dir.rstrip("/")),
        "namespace": prov.NS_PREFLIGHT,
        "smoke_test": True,
        "full_data": False,
        "timing_basis": "SMOKE_TIMING_ONLY",
        "reconstructed_from_legacy": True,
        "classification_evidence": ev,
        "note": ("Migrated T4 preflight evidence. Architecture/CUDA plumbing proof "
                 "only -- not a scientific result, not a timing basis, and not "
                 "adoptable: the identity hashes are absent by design."),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=os.environ.get("DRIVE_ROOT"), required=False)
    ap.add_argument("--apply", action="store_true", help="without this, nothing is written")
    args = ap.parse_args()
    root = args.drive_root
    if not root:
        raise SystemExit("--drive-root is required (or set DRIVE_ROOT)")
    if not os.path.isdir(root):
        raise SystemExit(f"not a directory: {root}")

    legacy_ckpts = os.path.join(root, "checkpoints")
    legacy_logs = os.path.join(root, "logs")
    quarantine = os.path.join(root, "quarantine")
    archive = os.path.join(root, "_legacy_pre_namespace")

    runs = sorted(d for d in os.listdir(legacy_ckpts)) if os.path.isdir(legacy_ckpts) else []
    runs = [d for d in runs if os.path.isdir(os.path.join(legacy_ckpts, d))]

    print("=" * 78)
    print(f"{'MIGRATION' if args.apply else 'DRY RUN -- nothing will be written'}: {root}")
    print("=" * 78)
    if not runs and not os.path.isdir(legacy_ckpts):
        print("no legacy checkpoints/ directory -- already migrated or a fresh Drive root")

    report = {"drive_root": root, "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "applied": bool(args.apply), "runs": []}

    for rid in runs:
        src = os.path.join(legacy_ckpts, rid)
        ev = classify(src)
        dest_ns = ev["verdict"]
        dest = (os.path.join(quarantine, rid) if dest_ns == "quarantine"
                else os.path.join(root, dest_ns, "checkpoints", rid))
        log_src = os.path.join(legacy_logs, f"{rid}.log")
        log_dest = (os.path.join(quarantine, f"{rid}.log") if dest_ns == "quarantine"
                    else os.path.join(root, dest_ns, "logs", f"{rid}.log"))
        print(f"\n{rid:<6} -> {dest_ns}")
        print(f"       {ev.get('why', '')}")
        print(f"       {src}\n    -> {dest}")
        if os.path.exists(log_src):
            print(f"       log -> {log_dest}")
        report["runs"].append({"id": rid, "verdict": dest_ns, "evidence": ev,
                               "dest": dest, "log_dest": log_dest})

        if args.apply:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.exists(dest):
                print(f"       SKIP copy: {dest} already exists (not overwriting)")
            else:
                shutil.copytree(src, dest)
            if dest_ns == prov.NS_PREFLIGHT:
                stamp_preflight(dest, ev)
            if os.path.exists(log_src):
                os.makedirs(os.path.dirname(log_dest), exist_ok=True)
                if not os.path.exists(log_dest):
                    shutil.copy2(log_src, log_dest)

    if args.apply:
        os.makedirs(archive, exist_ok=True)
        for sub in LEGACY_SUBDIRS:
            p = os.path.join(root, sub)
            if os.path.isdir(p):
                target = os.path.join(archive, sub)
                if os.path.exists(target):
                    target += f".{int(time.time())}"
                shutil.move(p, target)
                print(f"\narchived legacy {sub}/ -> {target}")
        os.makedirs(os.path.join(root, prov.NS_PREFLIGHT, "manifest"), exist_ok=True)
        rp = os.path.join(root, prov.NS_PREFLIGHT, "manifest", "migration_report.json")
        with open(rp, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"report -> {rp}")
    else:
        print("\n(dry run) re-run with --apply to perform the migration")

    print("\n" + "=" * 78)
    print("legacy artifacts are COPIED, never deleted; the flat tree is archived at")
    print(f"  {archive}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
