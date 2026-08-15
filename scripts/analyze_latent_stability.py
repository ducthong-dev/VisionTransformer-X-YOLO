#!/usr/bin/env python
"""Latent robustness analysis (Phase 7) -- evidence for "noise-resilient latent features".

Reviewer #10 challenges that phrase. This script measures it directly: for each
clean image x and its frozen corrupted counterpart x', it extracts the
representation BEFORE the auto-encoder and the AE latent, then compares how far
each drifts.

    D_pre = || z_pre(x) - z_pre(x') ||_2
    D_ae  = || z_ae(x)  - z_ae(x')  ||_2

Distances are additionally reported normalised by the clean representation's norm,
because z_pre and z_ae have different dimensionalities and raw magnitudes are not
comparable across them. Cosine similarity, intra/inter-class distance and
silhouette score are reported alongside.

    python scripts/analyze_latent_stability.py --run results/ablation/A5_aetfpe_full \
        --corruption pepper/030

This is empirical representation analysis. It does not prove superiority; it
shows whether the representation moves less under corruption while remaining
class-separable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import pick_device, resolve_roots  # noqa: E402
from aetfpe.data import PairedCorruptionDataset  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.seeding import seed_everything  # noqa: E402


def flat(t: torch.Tensor) -> np.ndarray:
    return t.reshape(t.shape[0], -1).float().cpu().numpy()


def drift_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.linalg.norm(a - b, axis=1)
    na = np.linalg.norm(a, axis=1) + 1e-12
    rel = d / na
    cos = (a * b).sum(1) / (na * (np.linalg.norm(b, axis=1) + 1e-12))
    return {
        "l2_mean": float(d.mean()), "l2_median": float(np.median(d)), "l2_std": float(d.std()),
        "relative_l2_mean": float(rel.mean()), "relative_l2_median": float(np.median(rel)),
        "relative_l2_std": float(rel.std()),
        "cosine_mean": float(cos.mean()), "cosine_std": float(cos.std()),
    }


def separability(z: np.ndarray, y: np.ndarray) -> dict:
    out: dict = {}
    classes = np.unique(y)
    cents = np.stack([z[y == c].mean(0) for c in classes])
    intra = float(np.mean([np.linalg.norm(z[y == c] - cents[i], axis=1).mean()
                           for i, c in enumerate(classes)]))
    if len(classes) > 1:
        dm = np.linalg.norm(cents[:, None] - cents[None], axis=2)
        inter = float(dm[np.triu_indices(len(classes), 1)].mean())
    else:
        inter = float("nan")
    out["intra_class_distance"] = intra
    out["inter_class_distance"] = inter
    out["inter_over_intra"] = float(inter / intra) if intra > 0 else float("nan")
    try:
        from sklearn.metrics import silhouette_score

        if len(classes) > 1 and len(z) > len(classes):
            out["silhouette"] = float(silhouette_score(z, y))
    except Exception as exc:  # noqa: BLE001
        out["silhouette_error"] = str(exc)
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corruption-root", default=None, help="default: ${OUTPUT_ROOT}/corruptions")
    ap.add_argument("--corruption", default="pepper/030",
                    help="relative path under the corruption root, e.g. pepper/030")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-embeddings", action="store_true")
    args = ap.parse_args()

    args.corruption_root = args.corruption_root or os.path.join(
        resolve_roots()["OUTPUT_ROOT"], "corruptions")
    ckpt = torch.load(os.path.join(args.run, "checkpoint.pt"), map_location="cpu")
    cfg, classes = ckpt["cfg"], ckpt["classes"]
    seed_everything(int((cfg.get("protocol") or {}).get("seed", 0)))
    device = pick_device(args.device)

    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    if not model.cfg.use_ae:
        print(f"NOTE: {cfg.get('name')} has no auto-encoder; only the pre-AE "
              f"representation will be reported.")

    clean_root = os.path.join(args.corruption_root, "clean")
    corrupt_root = os.path.join(args.corruption_root, args.corruption)
    ds = PairedCorruptionDataset(clean_root, corrupt_root, classes,
                                 int((cfg.get("protocol") or {}).get("img_size", 224)))
    if len(ds) == 0:
        print(f"no paired images between {clean_root} and {corrupt_root}")
        return 1
    if args.max_images and len(ds) > args.max_images:
        ds.items = ds.items[: args.max_images]

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    pre_c, pre_x, lat_c, lat_x, ys = [], [], [], [], []
    for clean, corrupt, y, _ in dl:
        _, p_c = model.frontend(clean.to(device), return_parts=True)
        _, p_x = model.frontend(corrupt.to(device), return_parts=True)
        pre_c.append(flat(p_c["pre_ae"]))
        pre_x.append(flat(p_x["pre_ae"]))
        if p_c["latent"] is not None:
            lat_c.append(flat(p_c["latent"]))
            lat_x.append(flat(p_x["latent"]))
        ys.append(y.numpy())

    y = np.concatenate(ys)
    pre_c, pre_x = np.concatenate(pre_c), np.concatenate(pre_x)

    result = {
        "run": args.run,
        "name": cfg.get("name"),
        "corruption": args.corruption,
        "num_images": int(len(y)),
        "pre_ae": {
            "dim": int(pre_c.shape[1]),
            "drift": drift_stats(pre_c, pre_x),
            "separability_clean": separability(pre_c, y),
            "separability_corrupted": separability(pre_x, y),
        },
    }

    if lat_c:
        lat_c, lat_x = np.concatenate(lat_c), np.concatenate(lat_x)
        result["ae_latent"] = {
            "dim": int(lat_c.shape[1]),
            "drift": drift_stats(lat_c, lat_x),
            "separability_clean": separability(lat_c, y),
            "separability_corrupted": separability(lat_x, y),
        }
        rp = result["pre_ae"]["drift"]["relative_l2_mean"]
        ra = result["ae_latent"]["drift"]["relative_l2_mean"]
        result["relative_drift_ratio_ae_over_pre"] = float(ra / rp) if rp > 0 else float("nan")
        result["interpretation"] = (
            "AE latent drifts LESS than the pre-AE representation"
            if ra < rp else
            "AE latent drifts MORE than the pre-AE representation"
        )

    out_dir = args.out or os.path.join(args.run, "latent")
    os.makedirs(out_dir, exist_ok=True)
    tag = args.corruption.replace("/", "_")
    with open(os.path.join(out_dir, f"latent_stability_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    if args.save_embeddings:
        np.savez_compressed(
            os.path.join(out_dir, f"embeddings_{tag}.npz"),
            pre_clean=pre_c, pre_corrupt=pre_x, labels=y,
            **({"ae_clean": lat_c, "ae_corrupt": lat_x} if lat_c is not None and len(lat_c) else {}),
        )

    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
