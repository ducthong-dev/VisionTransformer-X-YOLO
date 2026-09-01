#!/usr/bin/env python
"""Evaluate verified checkpoints on the Controlled Synthetic Corruption Benchmark.

    python scripts/evaluate_controlled_corruptions.py --device mps
    python scripts/evaluate_controlled_corruptions.py --runs A5 D1

Deliberately a SEPARATE script writing to a SEPARATE output tree
(`results/evaluation_controlled/`) from `scripts/evaluate_distributions.py`
(`results/evaluation/`). The two benchmarks answer different questions and must never
be merged into one unexplained average:

  A. Clean / Easy / Moderate / Hard  -> synthetic AUGMENTATION robustness
     (mixes photometric corruption with geometric augmentation; see
      docs/CORRUPTION_PROTOCOL.md -- ~84 % of `hard` is flipped or rotated)

  B. this benchmark                  -> targeted CORRUPTION / NOISE robustness
     (6 non-geometric families x mild/moderate/severe, deterministic per sample)

Corruption is applied on the fly from the frozen specification, after the same
Resize((224,224)) the clean evaluation uses, so a model's `clean` row here is exactly
its `clean` row there. Every corrupted pixel is a pure function of
(global_seed, relative_path, family, severity).

Inference only. Never trains, tunes, or selects on these images. Fully resumable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import environment_info, pick_device  # noqa: E402
from aetfpe.corruptions import apply_corruption  # noqa: E402
from aetfpe.data import list_classes, list_samples  # noqa: E402
from aetfpe.metrics import summarize  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.seeding import seed_everything, sha256_array  # noqa: E402

BENCHMARK = "Controlled Synthetic Corruption Benchmark"
NOT_CALLED = "real-world robustness benchmark"     # explicitly not this

PRIMARY = ["A0", "A5", "D1", "E5", "B2"]
SECONDARY_IF_CHEAP = ["F2", "F4"]
ALIASES = {"F3": "A3", "F5": "A5", "F5_clean": "D1"}

FIELDS = ["run_id", "benchmark", "family", "severity", "sample_id", "relative_path",
          "ground_truth_index", "ground_truth_class", "predicted_index", "predicted_class",
          "confidence", "top5_indices", "top5_scores", "correct_top1", "correct_top5"]


class ControlledCorruptionDataset(Dataset):
    """Clean test image -> Resize(224) -> deterministic corruption -> ToTensor."""

    def __init__(self, root, classes, family, severity, params, img_size, seed, hashes=None):
        self.root = root
        self.classes = classes
        self.samples = list_samples(root, classes)
        self.family, self.severity, self.params = family, severity, params
        self.seed = seed
        self.resize = transforms.Resize((img_size, img_size))
        self.to_tensor = transforms.ToTensor()
        self.hashes = hashes                      # optional integrity check vs the manifest
        self.mismatches: list[str] = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        rel = os.path.relpath(path, self.root)
        img = np.array(self.resize(Image.open(path).convert("RGB")))
        out = apply_corruption(img, self.family, self.params, rel, self.severity, seed=self.seed)
        if self.hashes is not None:
            want = self.hashes.get((rel, self.family, self.severity))
            if want and sha256_array(out) != want:
                self.mismatches.append(rel)
        return self.to_tensor(Image.fromarray(out)), label, rel


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_rows(path, rows):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


@torch.no_grad()
def run_one(model, ds, device, batch_size, workers, rid, family, severity, classes):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    rows, logits_all, y_all = [], [], []
    for x, y, rels in dl:
        logits = model(x.to(device)).float().cpu()
        logits_all.append(logits.numpy())
        y_all.append(y.numpy())
        probs = torch.softmax(logits, dim=1)
        s5, i5 = torch.topk(probs, k=min(5, probs.shape[1]), dim=1)
        for j, rel in enumerate(rels):
            gt, pi = int(y[j]), int(i5[j, 0])
            t5 = [int(v) for v in i5[j]]
            rows.append({
                "run_id": rid, "benchmark": "controlled", "family": family, "severity": severity,
                "sample_id": rel, "relative_path": rel,
                "ground_truth_index": gt, "ground_truth_class": classes[gt],
                "predicted_index": pi, "predicted_class": classes[pi],
                "confidence": round(float(s5[j, 0]), 6),
                "top5_indices": "|".join(str(v) for v in t5),
                "top5_scores": "|".join(f"{float(v):.6f}" for v in s5[j]),
                "correct_top1": int(pi == gt), "correct_top5": int(gt in t5),
            })
    return rows, summarize(np.concatenate(logits_all), np.concatenate(y_all), classes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="configs/controlled_corruptions.yaml")
    ap.add_argument("--frozen", default="results/controlled_corruptions")
    ap.add_argument("--campaign-root", default="results/campaign/scientific")
    ap.add_argument("--verification", default="results/campaign/checkpoint_verification.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="results/evaluation_controlled")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--check-hashes", action="store_true",
                    help="verify each corrupted image against the frozen manifest (slower)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import yaml
    if args.data_root is None:
        args.data_root = yaml.safe_load(open("configs/local.yaml"))["env"]["DATA_ROOT"]
    cfg = yaml.safe_load(open(args.spec))

    spec_json = os.path.join(args.frozen, "controlled_corruption_spec.json")
    if not os.path.exists(spec_json):
        raise SystemExit(
            f"ERROR: {spec_json} not found.\n"
            "The benchmark specification must be frozen first:\n"
            "  python scripts/generate_controlled_corruptions.py --hashes")
    frozen = json.load(open(spec_json))
    if frozen["spec_sha256"] != sha256_file(args.spec):
        raise SystemExit("ERROR: configs/controlled_corruptions.yaml has changed since the "
                         "benchmark was frozen. Refusing to evaluate against a moved target.")

    if not os.path.exists(args.verification):
        raise SystemExit(f"ERROR: {args.verification} not found. Verify checkpoints first.")
    ver = json.load(open(args.verification))
    accepted = set(ver.get("accepted") or [])
    if not accepted:
        raise SystemExit("ERROR: no ACCEPTED checkpoints. Nothing to evaluate.")

    order = args.runs or [r for r in PRIMARY + SECONDARY_IF_CHEAP if r in accepted]
    for r in order:
        if r in ALIASES:
            raise SystemExit(f"ERROR: {r} is an alias of {ALIASES[r]}; evaluate the physical run.")
        if r not in accepted:
            raise SystemExit(f"ERROR: {r} is not ACCEPTED. Refusing to evaluate it.")

    hashes = None
    if args.check_hashes:
        mp = os.path.join(args.frozen, frozen["manifest"])
        hashes = {}
        with gzip.open(mp, "rt") as fh:
            for r in csv.DictReader(fh):
                if r["pixel_sha256"]:
                    hashes[(r["relative_path"], r["family"], r["severity"])] = r["pixel_sha256"]
        print(f"loaded {len(hashes)} frozen pixel hashes for integrity checking")

    root = os.path.join(args.data_root, cfg["source_split"])
    classes = list_classes(root)
    size = int(cfg["image_size"])
    seed = int(cfg["global_seed"])
    device = pick_device(args.device)
    steps = [("clean", "none", {})] + [(f, s, dict(p))
                                       for f, sv in cfg["corruptions"].items()
                                       for s, p in sv.items()]
    os.makedirs(args.out, exist_ok=True)

    print(f"benchmark : {BENCHMARK}")
    print(f"            (never called a '{NOT_CALLED}')")
    print(f"device={device}  runs={order}  distributions={len(steps)}  images={len(classes) and ''}\n")

    for rid in order:
        run_out = os.path.join(args.out, rid)
        os.makedirs(run_out, exist_ok=True)
        pending = [(f, s, p) for f, s, p in steps
                   if args.force or not os.path.exists(os.path.join(run_out, f"predictions_{f}_{s}.csv.gz"))]
        if not pending:
            print(f"[{rid}] complete -- skipping (resume)")
            continue

        ck = torch.load(os.path.join(args.campaign_root, rid, "checkpoint.pt"),
                        map_location="cpu", weights_only=False)
        mcfg, ck_classes = ck["cfg"], list(ck["classes"])
        if ck_classes != classes:
            raise SystemExit(f"ERROR: {rid} class ordering differs from the dataset.")
        seed_everything(int((mcfg.get("protocol") or {}).get("seed", 0)))
        model = build_model(mcfg["model"])
        model.load_state_dict(ck["model"], strict=True)
        model = model.to(device).eval()
        print(f"[{rid}] epoch {ck['epoch']} val_top1={float(ck['val_top1']):.7f}  "
              f"pending={len(pending)}/{len(steps)}")

        sp = os.path.join(run_out, "eval_controlled.json")
        meta = json.load(open(sp)) if os.path.exists(sp) else {
            "run_id": rid, "benchmark": BENCHMARK,
            "benchmark_kind": "targeted corruption / noise robustness (non-geometric)",
            "separate_from": ("Clean/Easy/Moderate/Hard = synthetic augmentation robustness; "
                              "the two are reported separately and never averaged together"),
            "frozen_spec": frozen, "device": device, "batch_size": args.batch_size,
            "environment": environment_info(),
            "selected_epoch": int(ck["epoch"]), "selected_val_top1": float(ck["val_top1"]),
            "seed_variability": "UNAVAILABLE (single training seed)",
            "distributions": {},
        }

        for family, sev, params in pending:
            t0 = time.time()
            if family == "clean":
                ds = ControlledCorruptionDataset(root, classes, "clean", "none", {}, size, seed)
            else:
                ds = ControlledCorruptionDataset(root, classes, family, sev, params, size, seed,
                                                 hashes=hashes)
            rows, agg = run_one(model, ds, device, args.batch_size, args.num_workers,
                                rid, family, sev, classes)
            key = f"{family}_{sev}"
            p = os.path.join(run_out, f"predictions_{key}.csv.gz")
            write_rows(p, rows)
            with open(os.path.join(run_out, f"aggregate_{key}.json"), "w") as fh:
                json.dump(agg, fh, indent=2)
            meta["distributions"][key] = {
                "family": family, "severity": sev, "parameters": params,
                "display_name": cfg["display_names"].get(family, family),
                "num_images": agg["overall"]["num_images"],
                "top1": agg["overall"]["top1"], "top5": agg["overall"]["top5"],
                "predictions": os.path.basename(p),
                "predictions_sha256": sha256_file(p),
                "hash_mismatches": len(ds.mismatches),
                "seconds": round(time.time() - t0, 1),
            }
            with open(sp, "w") as fh:
                json.dump(meta, fh, indent=2, default=str)
            warn = f"  !! {len(ds.mismatches)} PIXEL HASH MISMATCHES" if ds.mismatches else ""
            print(f"    {family:15s} {sev:8s} top1={agg['overall']['top1']:.4f} "
                  f"({meta['distributions'][key]['seconds']}s){warn}")
        del model

    print(f"\nwrote {args.out}/<run>/predictions_<family>_<severity>.csv.gz")
    print(f"NOTE: this is the {BENCHMARK}. Report it separately from Clean/Easy/Moderate/Hard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
