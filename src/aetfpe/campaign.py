"""Campaign engine for the one-day Major Revision training run.

Owns the experiment matrix, the model-size filter, the priority tiers, and a
crash-durable manifest on Google Drive. `Multimedia_Reproduce.ipynb` is a thin
front-end over this module so the logic can be tested off Colab.

Design rules:
  * Drive is the source of truth. /content is scratch.
  * A COMPLETED run is never re-run without force=True.
  * Nothing here changes training semantics -- it only decides what to launch,
    in what order, and where the artifacts land.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default filter. > this many TOTAL parameters is skipped unless explicitly forced.
MAX_TRAIN_PARAMS = 20_000_000

PENDING, RUNNING, COMPLETED, FAILED = "PENDING", "RUNNING", "COMPLETED", "FAILED"
SKIPPED_SIZE, SKIPPED_REUSE, SKIPPED_TIME = "SKIPPED_SIZE", "SKIPPED_REUSE", "SKIPPED_TIME"
TERMINAL_OK = (COMPLETED, SKIPPED_SIZE, SKIPPED_REUSE)

# --------------------------------------------------------------------------- #
# The matrix. `reuse_of` marks a logical experiment satisfied by another run's
# checkpoint -- verified against scripts/print_run_matrix.py, which derives
# duplicates from the config signature rather than from these annotations.
# --------------------------------------------------------------------------- #
EXPERIMENTS = [
    # -- P0: the scientific minimum -----------------------------------------
    dict(id="A0", config="configs/baseline_rgb.yaml", priority="P0", group="B",
         title="YOLOv8n-cls RGB baseline",
         purpose="Fair Protocol-B reference. Nothing is interpretable without it.",
         reviewer="#10.7 fairness; every accuracy comparison"),
    dict(id="E5", config="configs/efficient/E5_efficient_c2_28.yaml", priority="P0", group="E",
         title="Efficient AE-TFPE (C2-28)",
         purpose="The Major-Revision contribution. Main candidate.",
         reviewer="#10.6 efficiency; the new method"),
    dict(id="M1", config="configs/mech_legacy_lut.yaml", priority="P0", group="M",
         title="Legacy LUT control",
         purpose="The historical transform was a zero-parameter LUT. If it matches "
                 "the method, the contribution is the LUT, not the architecture.",
         reviewer="#10.4 mechanism attribution"),
    dict(id="M2", config="configs/mech_photometric.yaml", priority="P0", group="M",
         title="Photometric (gamma) control",
         purpose="Rules out a plain contrast/gamma shift as the explanation.",
         reviewer="#10.4 mechanism attribution"),
    dict(id="M3", config="configs/mech_aug_control.yaml", priority="P0", group="M",
         title="Augmentation control",
         purpose="Rules out extra input variability as the explanation.",
         reviewer="#11 robustness attribution"),
    dict(id="E3", config="configs/efficient/E3_efficient_pe_tf_no_ae.yaml", priority="P0", group="E",
         title="Efficient PE+TF, no AE",
         purpose="Closest available 'AE removed' control for E5. CONFOUND: also "
                 "changes the fusion space (grid -> image); no cleaner control exists.",
         reviewer="#10.4 AE contribution"),

    # -- P1: high value ------------------------------------------------------
    dict(id="A1", config="configs/pe_only.yaml", priority="P1", group="A",
         title="PE-only",
         purpose="PE contribution. Serves BOTH method families: with no TF branch "
                 "the Original and Efficient variants are the same model.",
         reviewer="#12 component contribution"),
    dict(id="A4", config="configs/rgb_ae.yaml", priority="P1", group="A",
         title="RGB + image-space AE",
         purpose="Isolates the auto-encoder from the fusion, Original side.",
         reviewer="#10.4 AE contribution"),
    dict(id="E7", config="configs/efficient/E7_efficient_c2_7.yaml", priority="P1", group="E",
         title="Efficient AE-TFPE (C2-7)",
         purpose="Spatial-resolution control for E5: 7x7 vs 28x28 grid.",
         reviewer="#10.6 architecture justification"),
    dict(id="B2", config="configs/baseline_efficientnet_b0.yaml", priority="P1", group="B",
         title="EfficientNet-B0 baseline",
         purpose="Independent lightweight external baseline under the same protocol.",
         reviewer="#10.7 fair external comparison"),

    # -- P2: optional --------------------------------------------------------
    dict(id="A2", config="configs/tf_only.yaml", priority="P2", group="A",
         title="TF-only (ViT-B/16)",
         purpose="Transformer contribution, Original side.",
         reviewer="#12 component contribution"),
    dict(id="A3", config="configs/pe_tf_no_ae.yaml", priority="P2", group="A",
         title="PE+TF, no AE (= F3)",
         purpose="Original-side no-AE control. Also serves fusion arm F3.",
         reviewer="#12 component contribution + fusion"),
    dict(id="A5", config="configs/aetfpe_full.yaml", priority="P2", group="A",
         title="Original AE-TFPE full (= F5)",
         purpose="The reference formulation. Also serves fusion arm F5.",
         reviewer="#10.4 reference method"),
    dict(id="D1", config="configs/fusion_ae_standard.yaml", priority="P2", group="F",
         title="Original AE fusion, clean objective",
         purpose="Like-for-like AE-as-fusion test; A5-D1 isolates the denoising objective.",
         reviewer="#10.4 denoising claim; #12 fusion"),
    dict(id="F1", config="configs/fusion_add.yaml", priority="P2", group="F",
         title="Addition fusion", purpose="Fusion comparator.", reviewer="#12 fusion"),
    dict(id="F2", config="configs/fusion_concat.yaml", priority="P2", group="F",
         title="Concatenation fusion",
         purpose="Fusion comparator. NOTE: the only arm that modifies the classifier stem.",
         reviewer="#12 fusion"),
    dict(id="F4", config="configs/fusion_attention.yaml", priority="P2", group="F",
         title="Attention fusion", purpose="Fusion comparator.", reviewer="#12 fusion"),
    dict(id="B1", config="configs/baseline_resnet50.yaml", priority="P2", group="B",
         title="ResNet-50 baseline", purpose="External baseline.",
         reviewer="#10.7 fair external comparison"),
    dict(id="B3", config="configs/baseline_vit_b16.yaml", priority="P2", group="B",
         title="ViT-B/16 baseline", purpose="External baseline.",
         reviewer="#10.7 fair external comparison"),
]

# Logical experiments served by another run's checkpoint. Confirmed by
# scripts/print_run_matrix.py from the config signature.
REUSE = {
    "F3": ("A3", "identical config signature (alias F3_fusion_linear)"),
    "F5": ("A5", "identical config signature (alias F5_fusion_ae)"),
    "F5_clean": ("D1", "identical config signature (alias F5_fusion_ae_clean)"),
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def measure_params(config_path: str) -> tuple[int, int]:
    """(total, trainable). Built with pretrained=False so no weights download."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from aetfpe.config import load_experiment
    from aetfpe.models import build_model
    cfg = load_experiment(os.path.join(REPO_ROOT, config_path))
    m = dict(cfg["model"]); m["pretrained"] = False; m["vit_pretrained"] = False
    mod = build_model(m)
    tot = sum(p.numel() for p in mod.parameters())
    tr = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    del mod
    return int(tot), int(tr)


def build_matrix(max_params: int = MAX_TRAIN_PARAMS, force_ids=()) -> list[dict]:
    """Resolve parameters and apply the size filter. Nothing is silently dropped."""
    force = set(force_ids or ())
    rows = []
    for e in EXPERIMENTS:
        r = dict(e)
        try:
            r["params"], r["trainable_params"] = measure_params(e["config"])
        except Exception as exc:  # noqa: BLE001
            r["params"], r["trainable_params"] = -1, -1
            r["build_error"] = str(exc)
        big = r["params"] > max_params
        if big and e["id"] not in force:
            r["status"] = SKIPPED_SIZE
            frozen = r["params"] - r["trainable_params"]
            r["reason"] = (
                f"{r['params']:,} total params exceeds MAX_TRAIN_PARAMS={max_params:,}. "
                + (f"Only {r['trainable_params']:,} are trainable ({frozen:,} frozen), "
                   "so the cost is the frozen backbone's FORWARD pass, not optimiser work."
                   if frozen > 0 else
                   "All parameters are trainable, so this is a full-backward cost.")
            )
        else:
            r["status"] = PENDING
            r["reason"] = ("forced by the user despite exceeding the threshold" if big
                           else f"{r['params']:,} params within the {max_params:,} threshold")
        rows.append(r)
    return rows


class Campaign:
    """Crash-durable campaign state. Drive is the source of truth."""

    def __init__(self, drive_root: str, scratch_root: str = "/content/campaign_scratch"):
        self.drive_root = drive_root
        self.scratch_root = scratch_root
        self.campaign_dir = os.path.join(drive_root, "campaign")
        self.ckpt_dir = os.path.join(drive_root, "checkpoints")
        self.log_dir = os.path.join(drive_root, "logs")
        for d in (self.campaign_dir, self.ckpt_dir, self.log_dir,
                  os.path.join(drive_root, "configs"), os.path.join(drive_root, "completed"),
                  os.path.join(drive_root, "failed"), scratch_root):
            os.makedirs(d, exist_ok=True)
        self.manifest_path = os.path.join(self.campaign_dir, "campaign_manifest.json")
        self.summary_path = os.path.join(self.campaign_dir, "campaign_summary.csv")
        self.manifest = self._load()

    # ---------------- persistence ----------------
    def _load(self) -> dict:
        if os.path.exists(self.manifest_path):
            try:
                return json.load(open(self.manifest_path))
            except Exception:  # noqa: BLE001 - a torn write must not brick the campaign
                bad = self.manifest_path + f".corrupt.{int(time.time())}"
                shutil.copy2(self.manifest_path, bad)
                print(f"WARNING: manifest unreadable, preserved at {bad}; starting fresh")
        return {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": {}}

    def save(self) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.manifest, fh, indent=2, default=str)
        os.replace(tmp, self.manifest_path)
        self._write_summary()

    def _write_summary(self) -> None:
        cols = ["id", "model", "priority", "group", "params", "trainable_params", "status",
                "start_time", "end_time", "runtime_s", "best_val_top1", "best_val_top5",
                "checkpoint", "git_commit", "gpu", "reason"]
        tmp = self.summary_path + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for rid, r in sorted(self.manifest["runs"].items()):
                w.writerow({c: r.get(c, "") for c in cols})
        os.replace(tmp, self.summary_path)

    # ---------------- state ----------------
    def seed(self, matrix: list[dict], gpu: str = "") -> None:
        """Merge the matrix into the manifest without clobbering finished runs."""
        for row in matrix:
            rid = row["id"]
            cur = self.manifest["runs"].get(rid, {})
            if cur.get("status") in TERMINAL_OK:
                continue                                  # never downgrade a finished run
            self.manifest["runs"][rid] = {
                **cur,
                "id": rid, "model": row["title"], "priority": row["priority"],
                "group": row["group"], "config": row["config"],
                "params": row["params"], "trainable_params": row["trainable_params"],
                "status": row["status"], "reason": row["reason"],
                "purpose": row["purpose"], "reviewer": row["reviewer"],
                "gpu": gpu or cur.get("gpu", ""),
            }
        for logical, (source, why) in REUSE.items():
            self.manifest["runs"].setdefault(logical, {
                "id": logical, "model": f"reuses {source}", "priority": "-", "group": "F",
                "params": "", "trainable_params": "", "status": SKIPPED_REUSE,
                "reason": why, "checkpoint": f"(see {source})",
            })
        self.save()

    def adopt_existing(self, rid: str) -> bool:
        """Recognise a valid finished run already on Drive (e.g. a prior E5)."""
        d = os.path.join(self.ckpt_dir, rid)
        s = os.path.join(d, "train_summary.json")
        if not (os.path.exists(s) and os.path.exists(os.path.join(d, "checkpoint.pt"))):
            return False
        try:
            js = json.load(open(s))
        except Exception:  # noqa: BLE001
            return False
        done = js.get("epochs_completed")
        planned = js.get("epochs_planned") or js.get("epochs")
        if js.get("status") not in (None, "completed") or (done and planned and done < planned):
            return False
        r = self.manifest["runs"].setdefault(rid, {"id": rid})
        r.update(status=COMPLETED, best_val_top1=js.get("best_val_top1"),
                 checkpoint=os.path.join(d, "checkpoint.pt"),
                 runtime_s=js.get("train_seconds"),
                 reason="pre-existing completed run found on Drive; not retrained")
        self.save()
        return True

    def queue(self, priority: str) -> list[str]:
        return [rid for rid, r in sorted(self.manifest["runs"].items())
                if r.get("priority") == priority and r.get("status") in (PENDING, FAILED, RUNNING)]

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for r in self.manifest["runs"].values():
            out[r.get("status", "?")] = out.get(r.get("status", "?"), 0) + 1
        return out

    # ---------------- execution ----------------
    def run(self, rid: str, epochs=None, extra_args=(), force=False, gpu="",
            device: str = "cuda", checkpoint_every: int = 1) -> dict:
        r = self.manifest["runs"][rid]
        if r["status"] in TERMINAL_OK and not force:
            print(f"[{rid}] {r['status']} -- skipping ({r.get('reason','')})")
            return r

        scratch = os.path.join(self.scratch_root, rid)
        drive_out = os.path.join(self.ckpt_dir, rid)
        os.makedirs(drive_out, exist_ok=True)
        os.makedirs(scratch, exist_ok=True)

        # A new Colab runtime starts with empty scratch, so pull the resume point
        # back from Drive first. Without this, --resume would find nothing and the
        # run would silently restart from epoch 0.
        resumed_from = None
        for fn in ("last.pt", "metrics.csv", "train_summary.json"):
            src = os.path.join(drive_out, fn)
            if os.path.exists(src) and not os.path.exists(os.path.join(scratch, fn)):
                shutil.copy2(src, os.path.join(scratch, fn))
                if fn == "last.pt":
                    resumed_from = src
        if resumed_from:
            try:
                import torch
                ep = torch.load(resumed_from, map_location="cpu",
                                weights_only=False).get("epoch")
                print(f"[{rid}] restoring resume point from Drive: {ep} epochs already done")
            except Exception:  # noqa: BLE001
                print(f"[{rid}] restoring resume point from Drive")
        r.update(status=RUNNING, start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                 git_commit=git_commit(), gpu=gpu or r.get("gpu", ""),
                 checkpoint=os.path.join(drive_out, "checkpoint.pt"))
        self.save()

        cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "train.py"),
               "--config", os.path.join(REPO_ROOT, r["config"]),
               "--device", device, "--out", scratch, "--mirror", drive_out,
               "--resume", "--checkpoint-every", str(checkpoint_every)]
        if epochs:
            cmd += ["--epochs", str(epochs)]
        cmd += list(extra_args)

        t0 = time.time()
        log_path = os.path.join(self.log_dir, f"{rid}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"[{rid}] {' '.join(cmd[1:])}\n[{rid}] log -> {log_path}")
        with open(log_path, "a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {rid} =====\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            rc = proc.wait()

        r["runtime_s"] = round(time.time() - t0, 1)
        r["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        summ = os.path.join(drive_out, "train_summary.json")
        if rc == 0 and os.path.exists(summ):
            js = json.load(open(summ))
            r.update(status=COMPLETED, best_val_top1=js.get("best_val_top1"),
                     reason="completed")
            m = os.path.join(drive_out, "metrics.csv")
            if os.path.exists(m):
                rows = list(csv.DictReader(open(m)))
                j = [x for x in rows if x.get("stage") != "ae_warmup"] or rows
                v5 = [float(x["val_top5"]) for x in j if x.get("val_top5")]
                r["best_val_top5"] = max(v5) if v5 else ""
            open(os.path.join(self.drive_root, "completed", rid), "w").write(r["end_time"])
        else:
            r.update(status=FAILED, reason=f"exit code {rc}; see logs/{rid}.log")
            open(os.path.join(self.drive_root, "failed", rid), "w").write(r["reason"])
        self.save()
        print(f"[{rid}] {r['status']} in {r['runtime_s']}s  best_val_top1={r.get('best_val_top1')}")
        return r
