#!/usr/bin/env python
"""Prove the campaign survives losing /content, using ONLY preflight artifacts.

    python scripts/preflight_resume_test.py --drive-root <ROOT> --device cuda

The claim under test is the one the whole one-day campaign rests on: when Colab
disconnects, everything needed to continue is already on Drive, and continuing
means continuing -- not silently restarting at epoch 0, not losing the recorded
metrics, not losing the best checkpoint.

    1. train a disposable run far enough to write last.pt, mirrored to Drive
    2. delete the /content scratch directory outright
    3. restore from Drive and nothing else
    4. resume, and check:
         * it starts at the NEXT epoch
         * metrics.csv is contiguous 1..N with the pre-crash rows unchanged
         * best-so-far survived the crash
    5. negative control: ask to resume the same directory as a DIFFERENT
       experiment and check that it is refused

The probe run is `RESUME_PROBE` in the preflight namespace. No scientific
artifact is read or written; the script refuses to run if pointed at one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from aetfpe import provenance as prov  # noqa: E402

RUN_ID = "RESUME_PROBE"


def sh(cmd: list[str]) -> tuple[int, str]:
    print("+ " + " ".join(cmd[1:] if cmd and cmd[0].endswith("python") else cmd))
    p = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    buf = []
    for line in p.stdout:
        print(line, end="")
        buf.append(line)
    return p.wait(), "".join(buf)


def read_metrics(path: str) -> list[dict]:
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default=os.environ.get("DRIVE_ROOT"))
    ap.add_argument("--scratch-root", default="/content/campaign_scratch/preflight_resume")
    ap.add_argument("--config", default="configs/baseline_rgb.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--stop-after", type=int, default=2)
    ap.add_argument("--limit-per-class", type=int, default=2)
    args = ap.parse_args()
    if not args.drive_root:
        raise SystemExit("--drive-root is required (or set DRIVE_ROOT)")

    drive_out = os.path.join(args.drive_root, prov.NS_PREFLIGHT, "checkpoints", RUN_ID)
    manifest_dir = os.path.join(args.drive_root, prov.NS_PREFLIGHT, "manifest")
    # Belt and braces: this script deletes directories, so it verifies it is
    # inside the preflight namespace before it deletes anything.
    if os.path.sep + prov.NS_PREFLIGHT + os.path.sep not in drive_out + os.path.sep:
        raise SystemExit(f"refusing to run outside the preflight namespace: {drive_out}")
    scratch = os.path.join(args.scratch_root, RUN_ID)
    os.makedirs(manifest_dir, exist_ok=True)

    for d in (drive_out, scratch):
        if os.path.isdir(d):
            shutil.rmtree(d)
    os.makedirs(drive_out, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)

    base = [sys.executable, os.path.join(REPO_ROOT, "scripts", "train.py"),
            "--config", args.config, "--device", args.device,
            "--out", scratch, "--mirror", drive_out, "--resume",
            "--checkpoint-every", "1", "--run-id", RUN_ID,
            "--namespace", prov.NS_PREFLIGHT, "--smoke-test",
            "--limit-per-class", str(args.limit_per_class),
            "--epochs", str(args.epochs)]

    checks: dict = {}
    ev: dict = {"run_id": RUN_ID, "namespace": prov.NS_PREFLIGHT,
                "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "config": args.config, "device": args.device,
                "epochs": args.epochs, "stop_after": args.stop_after,
                "limit_per_class": args.limit_per_class,
                "drive_out": drive_out, "scratch": scratch}

    def check(key: str, passed: bool, detail: str) -> bool:
        checks[key] = {"passed": bool(passed), "detail": detail}
        print(f"  [{'PASS' if passed else 'FAIL'}] {key}: {detail}")
        return bool(passed)

    # ---- 1. run until the resume point exists ------------------------------
    print("\n=== STEP 1: train to epoch %d, mirroring to Drive ===" % args.stop_after)
    rc, _ = sh(base + ["--preflight-stop-after", str(args.stop_after)])
    check("phase1_exit_ok", rc == 0, f"exit code {rc}")

    before = read_metrics(os.path.join(drive_out, "metrics.csv"))
    summary_before = json.load(open(os.path.join(drive_out, "train_summary.json")))
    best_before = summary_before.get("best_val_top1")
    ev["best_before"] = best_before
    ev["epochs_before"] = [int(r["epoch"]) for r in before]
    check("phase1_wrote_drive_resume_point",
          all(os.path.exists(os.path.join(drive_out, f))
              for f in ("last.pt", "metrics.csv", "checkpoint.pt", "train_summary.json",
                        prov.PROVENANCE_FILE)),
          f"Drive holds {sorted(os.listdir(drive_out))}")
    check("phase1_not_marked_completed", summary_before.get("status") != "completed",
          f"status={summary_before.get('status')!r}")

    # ---- 2. lose /content --------------------------------------------------
    print("\n=== STEP 2: simulate the loss of /content scratch ===")
    shutil.rmtree(scratch)
    check("scratch_destroyed", not os.path.exists(scratch), f"removed {scratch}")

    # ---- 3. restore from Drive and nothing else ----------------------------
    print("\n=== STEP 3: restore ONLY from Drive ===")
    os.makedirs(scratch, exist_ok=True)
    restored = []
    for fn in sorted(os.listdir(drive_out)):
        src = os.path.join(drive_out, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(scratch, fn))
            restored.append(fn)
    ev["restored_from_drive"] = restored
    check("restored_from_drive_only", "last.pt" in restored,
          f"copied {len(restored)} file(s) from Drive: {restored}")

    # ---- 4. resume ---------------------------------------------------------
    print("\n=== STEP 4: resume ===")
    rc2, out2 = sh(base)
    check("phase2_exit_ok", rc2 == 0, f"exit code {rc2}")

    resumed_line = [l for l in out2.splitlines() if "RESUMED from last.pt" in l]
    ev["resume_line"] = resumed_line[0].strip() if resumed_line else ""
    check("announced_resume", bool(resumed_line),
          ev["resume_line"] or "trainer never announced a resume")

    next_ep = args.stop_after + 1
    ev["resumed_at_epoch"] = next_ep
    check("resumed_at_next_epoch",
          f"continuing at epoch {next_ep}" in "".join(resumed_line),
          f"expected to continue at epoch {next_ep}")
    first_new = [l for l in out2.splitlines() if l.strip().startswith("ep")]
    ev["first_epoch_line_after_resume"] = first_new[0].strip() if first_new else ""
    first_ep = None
    if first_new:                                   # "  ep  3/4 [joint] ..."
        try:
            first_ep = int(first_new[0].strip().split()[1].split("/")[0])
        except (IndexError, ValueError):
            first_ep = None
    check("first_trained_epoch_is_next", first_ep == next_ep,
          f"first epoch trained after the crash was {first_ep}, expected {next_ep}")

    after = read_metrics(os.path.join(drive_out, "metrics.csv"))
    epochs_after = [int(r["epoch"]) for r in after]
    ev["epochs_after"] = epochs_after
    ev["final_epoch"] = epochs_after[-1] if epochs_after else None
    check("metrics_contiguous", epochs_after == list(range(1, args.epochs + 1)),
          f"metrics.csv epochs = {epochs_after}, expected {list(range(1, args.epochs + 1))}")
    check("pre_crash_rows_unchanged",
          [{k: r[k] for k in ("epoch", "train_loss", "val_top1")} for r in before]
          == [{k: r[k] for k in ("epoch", "train_loss", "val_top1")}
              for r in after[:len(before)]],
          "the epochs recorded before the crash were not rewritten")

    summary_after = json.load(open(os.path.join(drive_out, "train_summary.json")))
    best_after = summary_after.get("best_val_top1")
    ev["best_after"] = best_after
    carried = f"best_val_top1={best_before:.4f}" in "".join(resumed_line) if resumed_line else False
    check("best_carried_into_resume", carried,
          f"resume announced the pre-crash best {best_before}")
    check("best_never_regressed", best_after >= best_before,
          f"best {best_before} -> {best_after}")
    ck = os.path.join(drive_out, "checkpoint.pt")
    check("best_checkpoint_present", os.path.exists(ck),
          f"{ck} exists ({os.path.getsize(ck) / 1e6:.1f} MB)" if os.path.exists(ck) else "missing")
    check("run_completed", summary_after.get("status") == "completed",
          f"status={summary_after.get('status')!r} after {summary_after.get('epochs_completed')} epochs")

    # ---- 5. negative control: a different experiment must be refused -------
    print("\n=== STEP 5: negative control -- resume this directory as a DIFFERENT run ===")
    rc3, out3 = sh(base[:-1] + [str(args.epochs + 2)])   # same dir, different epoch budget
    refused = rc3 != 0 and "REFUSED" in out3
    ev["negative_control_exit"] = rc3
    check("foreign_resume_refused", refused,
          "a mismatched epoch budget was refused" if refused
          else f"NOT refused (exit {rc3}) -- provenance guard is not working")

    ev["checks"] = checks
    ev["passed"] = all(c["passed"] for c in checks.values())
    path = os.path.join(manifest_dir, "resume_test.json")
    with open(path, "w") as fh:
        json.dump(ev, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"DRIVE-ONLY RESUME PREFLIGHT: {'PASS' if ev['passed'] else 'FAIL'}")
    print(f"evidence -> {path}")
    print("=" * 78)
    return 0 if ev["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
