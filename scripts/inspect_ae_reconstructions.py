#!/usr/bin/env python
"""Inspect AE reconstructions on a deterministic sample of VALIDATION images.

Answers the decoder-collapse question that per-epoch loss alone cannot: a decoder
that emits one constant image for every input can still show a falling
reconstruction loss.

VALIDATION ONLY. The split is hard-coded to `data['val_split']`; there is no code
path that can construct a test path, and the script aborts if the resolved
directory contains "test".

    python scripts/inspect_ae_reconstructions.py \
        --run "$OUTPUT_ROOT/validation/C2_28_clean_sanity" --n 8

Writes `ae_reconstructions.png` (original / reconstruction / absolute error) and
`ae_reconstruction_stats.json` into the run directory. Inference only -- no
training, no gradient, no weight update.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import load_experiment, pick_device  # noqa: E402
from aetfpe.data import LeafDataset, list_classes  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.seeding import seed_everything  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=8, help="validation images to sample")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stride", type=int, default=997,
                    help="deterministic spacing through the val index, for class spread")
    args = ap.parse_args()

    cfg_path = os.path.join(args.run, "config.yaml")
    ckpt_path = os.path.join(args.run, "checkpoint.pt")
    for p in (cfg_path, ckpt_path):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 2

    cfg = load_experiment(cfg_path) if cfg_path.endswith(".yaml") else {}
    d = cfg["data"]
    val_root = os.path.join(d["root"], d["val_split"])          # VALIDATION ONLY
    if "test" in os.path.normpath(val_root).lower().split(os.sep):
        print(f"ABORT: resolved split looks like a test path: {val_root}", file=sys.stderr)
        return 2
    if not os.path.isdir(val_root):
        print(f"ERROR: validation split not found: {val_root}", file=sys.stderr)
        return 2

    device = pick_device(args.device)
    seed_everything(0)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("cfg", cfg)["model"] if "cfg" in ckpt else cfg["model"])
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()
    if model.ae is None:
        print("ERROR: this run has no auto-encoder; nothing to inspect.", file=sys.stderr)
        return 2

    classes = ckpt.get("classes") or list_classes(val_root)
    ds = LeafDataset(val_root, classes, cfg["protocol"]["img_size"] if "protocol" in cfg else 224,
                     train=False)
    idx = [(i * args.stride) % len(ds) for i in range(args.n)]     # deterministic, spread out
    x = torch.stack([ds[i][0] for i in idx]).to(device)

    recon, parts = model.frontend(x, return_parts=True)
    err = (recon - x).abs()

    # Decoder-collapse test: if every reconstruction were the same image, the
    # spread across the batch would be ~0 while per-image spread stays non-zero.
    across = recon.std(dim=0).mean().item()
    within = recon.std(dim=(1, 2, 3)).mean().item()
    stats = {
        "run": args.run, "device": device, "n_images": args.n,
        "val_root": val_root, "sample_indices": idx,
        "input_range": [round(x.min().item(), 5), round(x.max().item(), 5)],
        "recon_range": [round(recon.min().item(), 5), round(recon.max().item(), 5)],
        "mae": round(err.mean().item(), 6),
        "rmse": round(err.pow(2).mean().sqrt().item(), 6),
        "per_image_mae": [round(v, 6) for v in err.flatten(1).mean(1).tolist()],
        "std_across_samples": round(across, 6),
        "std_within_sample": round(within, 6),
        "collapse_ratio": round(across / (within + 1e-12), 4),
        "latent_shape": list(parts["latent"].shape) if parts["latent"] is not None else None,
        "latent_mean": round(parts["latent"].mean().item(), 6) if parts["latent"] is not None else None,
        "latent_active_fraction": (
            round((parts["latent"] > 0.05).float().mean().item(), 4)
            if parts["latent"] is not None else None),
        "interpretation": (
            "collapse_ratio near 0 means every input maps to the same reconstruction "
            "(decoder collapse). A healthy decoder keeps it well above 0."),
    }

    out_json = os.path.join(args.run, "ae_reconstruction_stats.json")
    json.dump(stats, open(out_json, "w"), indent=2)
    print(json.dumps(stats, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = args.n
        fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.2))
        for j in range(n):
            for row, (img, title) in enumerate((
                    (x[j], "original"), (recon[j], "reconstruction"), (err[j], "|error|"))):
                ax = axes[row, j] if n > 1 else axes[row]
                ax.imshow(img.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(title, fontsize=9)
            (axes[0, j] if n > 1 else axes[0]).set_title(
                classes[ds[idx[j]][1]][:18], fontsize=7)
        fig.suptitle("AE reconstructions — VALIDATION split only", fontsize=10)
        fig.tight_layout()
        out_png = os.path.join(args.run, "ae_reconstructions.png")
        fig.savefig(out_png, dpi=120)
        print(f"\nwrote {out_png}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"\n(figure skipped: {exc})", file=sys.stderr)

    print(f"wrote {out_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
