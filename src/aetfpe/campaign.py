"""Campaign engine for the one-day Major Revision training run.

Owns the experiment matrix, the model-size filter, the priority tiers, and a
crash-durable manifest on Google Drive. `Multimedia_Reproduce.ipynb` is a thin
front-end over this module so the logic can be tested off Colab.

Design rules:
  * Drive is the source of truth. /content is scratch.
  * A COMPLETED run is never re-run without force=True.
  * Nothing here changes training semantics -- it only decides what to launch,
    in what order, and where the artifacts land.
  * **Preflight and scientific artifacts share no directory.** A smoke run and
    the scientific run of the same arm have the same ID and write the same file
    names, so the only durable defence is that they are never allowed to write
    to the same place:

        <drive_root>/preflight/{checkpoints,logs,manifest}
        <drive_root>/scientific/{checkpoints,logs,manifest,summaries}

    `Campaign(namespace="preflight")` is smoke-only; `namespace="scientific"` is
    full-data-only. Neither can address the other's tree.
  * **Adoption and resume are provenance-checked, not name-checked.** See
    `aetfpe.provenance`: run ID, config hash, protocol hash, smoke flag, epochs
    requested, per-class limits, full-data status and dataset hash must all
    match, or the artifact is refused loudly and left untouched.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time

from . import provenance as prov

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default filter. > this many TOTAL parameters is skipped unless explicitly forced.
MAX_TRAIN_PARAMS = 20_000_000

PENDING, RUNNING, COMPLETED, FAILED = "PENDING", "RUNNING", "COMPLETED", "FAILED"
SKIPPED_SIZE, SKIPPED_REUSE, SKIPPED_TIME = "SKIPPED_SIZE", "SKIPPED_REUSE", "SKIPPED_TIME"
REFUSED_PROVENANCE = "REFUSED_PROVENANCE"
TERMINAL_OK = (COMPLETED, SKIPPED_SIZE, SKIPPED_REUSE)

# The fusion/original-side arms that exceed MAX_TRAIN_PARAMS but are trained
# anyway, because the claims they carry cannot otherwise be made. A3 is here as
# well as A5/D1/F1/F2/F4: it is the physical run behind fusion arm F3, which is
# a logical reuse (see REUSE) and must never be trained a second time.
FORCED_FUSION_IDS = ("A5", "D1", "F1", "F2", "A3", "F4")

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
    """Crash-durable campaign state for ONE namespace. Drive is the source of truth.

    `namespace` is not a label -- it selects a physically separate artifact tree
    and fixes what this campaign is allowed to launch:

        preflight   smoke runs only  (a per-class limit is mandatory)
        scientific  full-data runs only (no limit, no smoke flag)

    A Campaign object cannot read or write the other namespace's checkpoints,
    logs or manifest, so a preflight checkpoint can never be adopted by the
    scientific campaign -- not because a check passed, but because the path does
    not exist in this object.
    """

    def __init__(self, drive_root: str, namespace: str = prov.NS_SCIENTIFIC,
                 scratch_root: str = "/content/campaign_scratch",
                 limit_per_class: int | None = None, epochs: int | None = None,
                 data_root: str | None = None):
        if namespace not in prov.NAMESPACES:
            raise ValueError(f"namespace must be one of {prov.NAMESPACES}, got {namespace!r}")
        self.drive_root = drive_root
        self.namespace = namespace
        self.smoke = namespace == prov.NS_PREFLIGHT
        if self.smoke and not limit_per_class:
            raise ValueError("the preflight namespace requires limit_per_class -- it "
                             "exists for disposable plumbing tests, not for full runs")
        if not self.smoke and limit_per_class:
            raise ValueError("the scientific namespace refuses limit_per_class: a "
                             "subset run is a preflight run and belongs in preflight/")
        self.limit_per_class = limit_per_class
        self.epochs = epochs
        self.data_root = data_root

        self.ns_root = os.path.join(drive_root, namespace)
        self.ckpt_dir = os.path.join(self.ns_root, "checkpoints")
        self.log_dir = os.path.join(self.ns_root, "logs")
        self.manifest_dir = os.path.join(self.ns_root, "manifest")
        self.summary_dir = (os.path.join(self.ns_root, "summaries") if not self.smoke
                            else self.manifest_dir)
        self.scratch_root = os.path.join(scratch_root, namespace)
        for d in (self.ckpt_dir, self.log_dir, self.manifest_dir, self.summary_dir,
                  os.path.join(self.ns_root, "configs"),
                  os.path.join(self.ns_root, "completed"),
                  os.path.join(self.ns_root, "failed"), self.scratch_root):
            os.makedirs(d, exist_ok=True)
        # A marker file so a human (or a later tool) reading a bare directory on
        # Drive can tell which half of the tree they are looking at.
        with open(os.path.join(self.ns_root, "NAMESPACE"), "w") as fh:
            fh.write(f"{namespace}\n"
                     f"{'SMOKE / PREFLIGHT ONLY -- not scientific evidence' if self.smoke else 'SCIENTIFIC -- full data only'}\n")

        self.manifest_path = os.path.join(self.manifest_dir, "campaign_manifest.json")
        self.summary_path = os.path.join(self.summary_dir, "campaign_summary.csv")
        self.manifest = self._load()
        self._ds_cache: dict = {}

    # ---------------- persistence ----------------
    def _load(self) -> dict:
        if os.path.exists(self.manifest_path):
            try:
                js = json.load(open(self.manifest_path))
            except Exception:  # noqa: BLE001 - a torn write must not brick the campaign
                bad = self.manifest_path + f".corrupt.{int(time.time())}"
                shutil.copy2(self.manifest_path, bad)
                print(f"WARNING: manifest unreadable, preserved at {bad}; starting fresh")
            else:
                found = js.get("namespace")
                if found and found != self.namespace:
                    raise RuntimeError(
                        f"manifest at {self.manifest_path} declares namespace {found!r} "
                        f"but was opened as {self.namespace!r}. Refusing to mix the two "
                        "artifact trees.")
                js["namespace"] = self.namespace
                return js
        return {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "namespace": self.namespace, "runs": {}}

    def save(self) -> None:
        self.manifest["namespace"] = self.namespace
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.manifest, fh, indent=2, default=str)
        os.replace(tmp, self.manifest_path)
        self._write_summary()

    def _write_summary(self) -> None:
        cols = ["id", "model", "priority", "group", "params", "trainable_params", "status",
                "namespace", "smoke_test", "timing_basis", "epochs_requested",
                "limit_per_class", "train_images", "start_time", "end_time", "runtime_s",
                "best_val_top1", "best_val_top5", "checkpoint", "git_commit", "gpu", "reason"]
        tmp = self.summary_path + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for rid, r in sorted(self.manifest["runs"].items()):
                w.writerow({c: r.get(c, "") for c in cols})
        os.replace(tmp, self.summary_path)

    # ---------------- provenance ----------------
    def _dataset_fingerprints(self, cfg: dict):
        """(train_fp, val_fp) for a config's split pair, cached per root."""
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from aetfpe.data import dataset_fingerprint, list_classes
        d = cfg["data"]
        root = self.data_root or d["root"]
        tr = os.path.join(root, d["train_split"])
        va = os.path.join(root, d["val_split"])
        key = (tr, va)
        if key not in self._ds_cache:
            classes = list_classes(tr)
            self._ds_cache[key] = (dataset_fingerprint(tr, classes),
                                   dataset_fingerprint(va, classes))
        return self._ds_cache[key]

    def expected_provenance(self, rid: str, epochs: int | None = None) -> dict:
        """The provenance record a valid artifact for `rid` must carry.

        Raises if it cannot be computed (e.g. the dataset is not mounted): an
        unverifiable artifact is refused, never adopted on trust.
        """
        sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
        from aetfpe.config import build_protocol, load_experiment
        r = self.manifest["runs"][rid]
        cfg = load_experiment(os.path.join(REPO_ROOT, r["config"]))
        protocol = build_protocol(cfg)
        ep = epochs if epochs is not None else (self.epochs or protocol.epochs)
        protocol.epochs = ep
        train_fp, val_fp = self._dataset_fingerprints(cfg)
        return prov.build(
            run_id=rid, namespace=self.namespace, smoke_test=self.smoke,
            config_path=r["config"], cfg=cfg, protocol=protocol, epochs_requested=ep,
            limit_per_class=self.limit_per_class,
            limit_train_per_class=None, limit_val_per_class=None,
            train_fp=train_fp, val_fp=val_fp)

    def _check_artifacts(self, rid: str, epochs: int | None, what: str) -> list[str]:
        """Mismatches between this run's requirements and what is on Drive."""
        d = os.path.join(self.ckpt_dir, rid)
        found = prov.load(d)
        try:
            expected = self.expected_provenance(rid, epochs)
        except Exception as exc:  # noqa: BLE001
            return [f"cannot compute the expected provenance ({exc}); "
                    "refusing rather than trusting the artifact"]
        return prov.compare(expected, found)

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
                "namespace": self.namespace,
                "gpu": gpu or cur.get("gpu", ""),
            }
        for logical, (source, why) in REUSE.items():
            self.manifest["runs"].setdefault(logical, {
                "id": logical, "model": f"reuses {source}", "priority": "-", "group": "F",
                "params": "", "trainable_params": "", "status": SKIPPED_REUSE,
                "namespace": self.namespace,
                "reason": why, "checkpoint": f"(see {source})",
            })
        self.save()

    def adopt_existing(self, rid: str, epochs: int | None = None, verbose: bool = True) -> bool:
        """Recognise a finished run already on Drive -- only if it is THIS run.

        The old version accepted any directory that held a checkpoint and a
        summary saying "completed". A 4-epoch smoke run satisfies both, which is
        how five plumbing tests came to be reported as completed science. It now
        has to prove it is the same experiment first.
        """
        if rid not in self.manifest["runs"]:
            return False
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

        mismatches = self._check_artifacts(rid, epochs or planned, "adoption")
        if mismatches:
            print(prov.refuse("adoption of an existing run", rid, d, mismatches))
            r = self.manifest["runs"][rid]
            if r.get("status") not in TERMINAL_OK:
                r.update(status=REFUSED_PROVENANCE,
                         reason="artifact on Drive failed provenance validation: "
                                + "; ".join(mismatches))
                self.save()
            return False

        rec = prov.load(d)
        r = self.manifest["runs"][rid]
        r.update(status=COMPLETED, best_val_top1=js.get("best_val_top1"),
                 checkpoint=os.path.join(d, "checkpoint.pt"),
                 runtime_s=js.get("train_seconds"),
                 namespace=self.namespace, smoke_test=rec.get("smoke_test"),
                 timing_basis=rec.get("timing_basis"),
                 epochs_requested=rec.get("epochs_requested"),
                 limit_per_class=rec.get("limit_per_class"),
                 train_images=js.get("train_images"),
                 reason="pre-existing completed run on Drive, provenance verified; not retrained")
        self.save()
        if verbose:
            print(f"ADOPTED {rid}: {prov.describe(rec)}")
        return True

    def revalidate(self) -> dict:
        """Demote every COMPLETED run that cannot prove it belongs to this namespace.

        This is the rebuild step: run it against a manifest inherited from before
        the namespace split, and anything that is really a smoke run drops back to
        PENDING with the reason recorded.
        """
        report = {"namespace": self.namespace, "kept": [], "demoted": []}
        for rid, r in sorted(self.manifest["runs"].items()):
            if r.get("status") != COMPLETED:
                continue
            d = os.path.join(self.ckpt_dir, rid)
            rec = prov.load(d)
            bad = []
            if not os.path.isdir(d):
                bad.append(f"no artifact directory in {self.namespace}/checkpoints")
            elif not rec:
                bad.append("artifact carries no provenance record")
            else:
                if rec.get("namespace") != self.namespace:
                    bad.append(f"artifact namespace {rec.get('namespace')!r}")
                if bool(rec.get("smoke_test")) != self.smoke:
                    bad.append(f"smoke_test={rec.get('smoke_test')} in a "
                               f"{self.namespace} campaign")
                if not self.smoke and not rec.get("full_data", False):
                    bad.append(f"not a full-data run (full_data="
                               f"{rec.get('full_data')}, limit_per_class="
                               f"{rec.get('limit_per_class')})")
            if bad:
                r.update(status=PENDING,
                         reason="DEMOTED by revalidate(): " + "; ".join(bad),
                         best_val_top1="", best_val_top5="", runtime_s="",
                         checkpoint="", timing_basis="", train_images="")
                report["demoted"].append({"id": rid, "why": bad})
            else:
                report["kept"].append(rid)
        self.save()
        path = os.path.join(self.manifest_dir, "revalidation_report.json")
        with open(path, "w") as fh:
            json.dump({**report, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=2)
        return report

    def queue(self, priority: str) -> list[str]:
        return [rid for rid, r in sorted(self.manifest["runs"].items())
                if r.get("priority") == priority and r.get("status") in (PENDING, FAILED, RUNNING)]

    def counts(self) -> dict:
        out: dict[str, int] = {}
        for r in self.manifest["runs"].values():
            out[r.get("status", "?")] = out.get(r.get("status", "?"), 0) + 1
        return out

    def completed_ids(self) -> list[str]:
        return sorted(rid for rid, r in self.manifest["runs"].items()
                      if r.get("status") == COMPLETED)

    # ---------------- execution ----------------
    def run(self, rid: str, epochs=None, extra_args=(), force=False, gpu="",
            device: str = "cuda", checkpoint_every: int = 1) -> dict:
        r = self.manifest["runs"][rid]
        if r["status"] in TERMINAL_OK and not force:
            print(f"[{rid}] {r['status']} -- skipping ({r.get('reason','')})")
            return r

        epochs = epochs if epochs is not None else self.epochs
        scratch = os.path.join(self.scratch_root, rid)
        drive_out = os.path.join(self.ckpt_dir, rid)
        os.makedirs(drive_out, exist_ok=True)
        os.makedirs(scratch, exist_ok=True)

        # Anything already on Drive under this ID must prove it is this run before
        # a single byte of it is copied into scratch. Restoring a foreign resume
        # point is the failure this check exists to prevent; the run is refused
        # rather than quietly restarted from epoch 0 on top of it.
        if os.path.exists(os.path.join(drive_out, "last.pt")):
            mismatches = self._check_artifacts(rid, epochs, "resume")
            if mismatches:
                print(prov.refuse("resume from Drive", rid, drive_out, mismatches))
                r.update(status=REFUSED_PROVENANCE, end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                         reason="Drive artifact failed provenance validation: "
                                + "; ".join(mismatches))
                self.save()
                return r

        # A new Colab runtime starts with empty scratch, so pull the resume point
        # back from Drive first. Without this, --resume would find nothing and the
        # run would silently restart from epoch 0.
        resumed_from = None
        for fn in ("last.pt", "metrics.csv", "train_summary.json", "checkpoint.pt",
                   "config.yaml", prov.PROVENANCE_FILE):
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
                 namespace=self.namespace, smoke_test=self.smoke,
                 epochs_requested=epochs, limit_per_class=self.limit_per_class,
                 checkpoint=os.path.join(drive_out, "checkpoint.pt"))
        self.save()

        cmd = [sys.executable, os.path.join(REPO_ROOT, "scripts", "train.py"),
               "--config", os.path.join(REPO_ROOT, r["config"]),
               "--device", device, "--out", scratch, "--mirror", drive_out,
               "--resume", "--checkpoint-every", str(checkpoint_every),
               "--run-id", rid, "--namespace", self.namespace]
        if self.smoke:
            cmd += ["--smoke-test", "--limit-per-class", str(self.limit_per_class)]
        if epochs:
            cmd += ["--epochs", str(epochs)]
        cmd += list(extra_args)

        t0 = time.time()
        log_path = os.path.join(self.log_dir, f"{rid}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"[{rid}] {' '.join(cmd[1:])}\n[{rid}] log -> {log_path}")
        with open(log_path, "a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {rid} "
                      f"[{self.namespace}] =====\n")
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
            r["timing_basis"] = js.get("timing_basis",
                                       "SMOKE_TIMING_ONLY" if self.smoke else "FULL_DATA")
            r["train_images"] = js.get("train_images")
            if js.get("status") == "completed":
                r.update(status=COMPLETED, best_val_top1=js.get("best_val_top1"),
                         reason="completed")
                m = os.path.join(drive_out, "metrics.csv")
                if os.path.exists(m):
                    rows = list(csv.DictReader(open(m)))
                    j = [x for x in rows if x.get("stage") != "ae_warmup"] or rows
                    v5 = [float(x["val_top5"]) for x in j if x.get("val_top5")]
                    r["best_val_top5"] = max(v5) if v5 else ""
                open(os.path.join(self.ns_root, "completed", rid), "w").write(r["end_time"])
            else:
                # e.g. an intentional preflight interruption: durable, but not done.
                r.update(status=PENDING, best_val_top1=js.get("best_val_top1"),
                         reason=f"stopped early with status {js.get('status')!r}; "
                                "resume point is on Drive")
        else:
            r.update(status=FAILED, reason=f"exit code {rc}; see {self.namespace}/logs/{rid}.log")
            open(os.path.join(self.ns_root, "failed", rid), "w").write(r["reason"])
        self.save()
        print(f"[{rid}] {r['status']} in {r['runtime_s']}s  best_val_top1={r.get('best_val_top1')}"
              + ("   <<< SMOKE TIMING ONLY -- not a full-data basis >>>" if self.smoke else ""))
        return r

    # ---------------- timing ----------------
    def measured_epoch_seconds(self, rid: str) -> tuple[float, str]:
        """(mean epoch seconds, timing basis). The basis travels with the number."""
        m = os.path.join(self.ckpt_dir, rid, "metrics.csv")
        rows = list(csv.DictReader(open(m))) if os.path.exists(m) else []
        if not rows:
            return 0.0, "NONE"
        mean = sum(float(x["seconds"]) for x in rows) / len(rows)
        basis = (prov.load(os.path.join(self.ckpt_dir, rid)) or {}).get(
            "timing_basis", "SMOKE_TIMING_ONLY" if self.smoke else "FULL_DATA")
        return mean, basis

    def assert_projectable(self, rid: str) -> float:
        """Mean epoch seconds, or a loud refusal if the measurement is a smoke number.

        156 training images is 0.4% of the 38,584 the scientific run sees. A
        projection built on it is not conservative, not optimistic and not an
        estimate -- it is a different quantity wearing the same unit.
        """
        mean, basis = self.measured_epoch_seconds(rid)
        if basis != "FULL_DATA":
            raise prov.ProvenanceMismatch(
                f"\n{'=' * 78}\nREFUSED: full-data projection from {rid}\n{'=' * 78}\n"
                f"measured epoch time carries timing_basis={basis!r}.\n"
                "Smoke epochs run on a per-class subset, so their wall-clock says nothing\n"
                "about full-data runtime and must never enter the campaign cost model or\n"
                "the forced-tier gate. Measure on full data, on the target GPU.\n"
                f"{'=' * 78}\n")
        return mean
