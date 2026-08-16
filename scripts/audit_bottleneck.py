#!/usr/bin/env python
"""Stage C2-2 -- information-preservation and PE-branch audit for candidate C2.

Trains nothing. Runs inference over a deterministic validation subset and asks two
questions that can be answered honestly WITHOUT a trained auto-encoder:

  1. Does the 7x7 feature-space bottleneck still separate disease classes?
  2. Does the PE branch carry image-dependent information, or is it a near-constant
     positional bias?

WHY NOT RECONSTRUCTION METRICS. The audit brief asks for MSE/PSNR/SSIM on the AE
reconstruction. At this stage the auto-encoder is randomly initialised: its decoder
emits a near-constant ~0.497 grey field whose across-image variation is ~0.017.
Reconstruction metrics computed on it would measure the random initialisation, not
the architecture, and would be worse than useless -- they would look like evidence.
They are therefore NOT reported. Two valid substitutes are reported instead:

  * SPATIAL-RESOLUTION FLOOR: downsample->upsample the image through a 7x7 grid
    (and 28x28, C0's grid) and measure MSE/PSNR/SSIM. This is what a purely
    *spatial* 7x7 code could reconstruct. Because the real latent carries 64
    channels (3,136 dims vs a 3-channel thumbnail's 147), this is a pessimistic
    FLOOR, not a ceiling -- it shows what 7x7 spatial resolution costs, and no more.

  * CLASS-SEPARABILITY AT THE BOTTLENECK: the decoder is a deterministic function
    of the latent, so it can only distinguish images whose latents differ. Measuring
    separability of the latent therefore bounds what any decoder could preserve.
    The AE encoder here is a random 1x1 projection 323->64; by Johnson-Lindenstrauss
    such a projection approximately preserves pairwise distances, so separability
    measured on it is a LOWER bound on what a trained encoder would achieve.

    python scripts/audit_bottleneck.py --per-class 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.config import environment_info, load_experiment, pick_device, resolve_roots  # noqa: E402
from aetfpe.data import IMG_EXT, list_classes  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.seeding import seed_everything  # noqa: E402

C2 = dict(use_pe=True, use_tf=True, use_ae=True, fusion="linear",
          tf_backbone="mobilevit_xxs", ae_space="feature")


# ----------------------------------------------------------------- metrics -- #

def _gauss_win(size=11, sigma=1.5):
    c = np.arange(size) - size // 2
    g = np.exp(-(c ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def ssim(a: np.ndarray, b: np.ndarray, L=1.0) -> float:
    """Mean SSIM over channels. a, b float HWC in [0,1]. Gaussian 11x11, sigma 1.5."""
    from scipy.signal import fftconvolve

    w = _gauss_win()
    C1, C2_ = (0.01 * L) ** 2, (0.03 * L) ** 2
    vals = []
    for c in range(a.shape[2]):
        x, y = a[:, :, c], b[:, :, c]
        mx = fftconvolve(x, w, mode="valid")
        my = fftconvolve(y, w, mode="valid")
        mxx = fftconvolve(x * x, w, mode="valid") - mx * mx
        myy = fftconvolve(y * y, w, mode="valid") - my * my
        mxy = fftconvolve(x * y, w, mode="valid") - mx * my
        s = ((2 * mx * my + C1) * (2 * mxy + C2_)) / ((mx ** 2 + my ** 2 + C1) * (mxx + myy + C2_))
        vals.append(float(s.mean()))
    return float(np.mean(vals))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


# ------------------------------------------------------------- descriptors -- #

def feature_stats(feats: np.ndarray, name: str) -> dict:
    """feats: [N, D] float32. Statistics requested by the audit brief."""
    n = feats.shape[0]
    per_image_norm = np.linalg.norm(feats, axis=1)
    # across-sample variance: how much a given feature dimension varies BETWEEN images
    across = float(feats.var(axis=0).mean())
    # within-feature variance: how much dimensions vary WITHIN one image, averaged
    within = float(feats.var(axis=1).mean())

    normed = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    idx = np.random.default_rng(0).choice(n, size=min(n, 300), replace=False)
    cos = normed[idx] @ normed[idx].T
    iu = np.triu_indices(len(idx), 1)
    pairwise_cos = cos[iu]

    return {
        "name": name,
        "dims": int(feats.shape[1]),
        "feature_norm_mean": float(per_image_norm.mean()),
        "feature_norm_std": float(per_image_norm.std()),
        "across_sample_variance": across,
        "within_feature_variance": within,
        "across_over_within": float(across / within) if within > 0 else float("nan"),
        "cosine_across_images_mean": float(pairwise_cos.mean()),
        "cosine_across_images_std": float(pairwise_cos.std()),
        # 1 - cos measures how distinguishable two random images are in this space.
        # Near 0 => the representation is nearly the same for every image.
        "mean_angular_distinguishability": float(1.0 - pairwise_cos.mean()),
    }


def separability(feats: np.ndarray, labels: np.ndarray) -> dict:
    """Leave-one-out 1-NN accuracy + silhouette. Robust at small N, no training."""
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    X = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    out = {}
    try:
        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        scores = cross_val_score(knn, X, labels, cv=min(5, np.bincount(labels).min()))
        out["knn1_cv_accuracy"] = float(scores.mean())
        out["knn1_cv_std"] = float(scores.std())
    except Exception as exc:  # noqa: BLE001
        out["knn1_error"] = str(exc)
    try:
        out["silhouette"] = float(silhouette_score(X, labels, metric="cosine"))
    except Exception as exc:  # noqa: BLE001
        out["silhouette_error"] = str(exc)
    out["chance_accuracy"] = float(1.0 / len(np.unique(labels)))
    return out


# -------------------------------------------------------------------- main -- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/_base.yaml")
    ap.add_argument("--per-class", type=int, default=12,
                    help="deterministic images per class (12 x 39 = 468, in the 200-500 band)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    seed_everything(0)
    device = pick_device(args.device)
    out_dir = args.out_dir or os.path.join(resolve_roots()["OUTPUT_ROOT"], "architecture_v2", "bottleneck_audit")
    os.makedirs(out_dir, exist_ok=True)

    d = load_experiment(args.config)["data"]
    train_root = os.path.join(d["root"], d["train_split"])
    val_root = os.path.join(d["root"], d["val_split"])
    classes = list_classes(train_root)

    # Deterministic subset: sorted classes, sorted filenames, first K of each.
    samples = []
    for ci, c in enumerate(classes):
        cd = os.path.join(val_root, c)
        if not os.path.isdir(cd):
            continue
        files = sorted(f for f in os.listdir(cd) if f.endswith(IMG_EXT))[: args.per_class]
        samples += [(os.path.join(cd, f), ci, c) for f in files]
    print(f"deterministic subset: {len(samples)} images over {len(classes)} classes "
          f"({args.per_class}/class), val split")

    model = build_model({**C2, "num_classes": len(classes), "pretrained": False,
                         "vit_pretrained": True}).to(device).eval()
    pe_mod, tf_mod, ae = model.pe, model.tf, model.ae
    gamma = model.cfg.pe_gamma
    grid = 7

    F_RGB, F_PE, F_TF, Z_AE, LAB = [], [], [], [], []
    PE_DELTA, TF_REL_DELTA = [], []
    # Spatial floors for every candidate grid, on the SAME images.
    GRIDS = (7, 14, 28)
    floors_acc = {g: [] for g in GRIDS}

    with torch.no_grad():
        for i in range(0, len(samples), args.batch_size):
            batch = samples[i: i + args.batch_size]
            imgs = np.stack([
                np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC),
                           dtype=np.float32) / 255.0
                for p, _, _ in batch])
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)

            pe = pe_mod(x)                                   # clamp(x + gamma*PE)
            pe_grid = F.adaptive_avg_pool2d(pe, (grid, grid))     # F_PE  [B,3,7,7]
            raw_grid = F.adaptive_avg_pool2d(x, (grid, grid))     # same, WITHOUT PE
            tf_pe = tf_mod.forward_features(pe)                   # F_TF  [B,320,7,7]
            tf_raw = tf_mod.forward_features(x)                   # TF without PE
            fused = torch.cat([pe_grid, tf_pe], dim=1)            # [B,323,7,7]
            z = ae.encode(fused)                                  # Z_AE  [B,64,7,7]

            # PE branch: how much of F_PE is image content vs the constant PE offset
            PE_DELTA.append((pe_grid - raw_grid).flatten(1).cpu().numpy())
            # PE influence on the transformer features, relative to their magnitude
            rel = ((tf_pe - tf_raw).flatten(1).norm(dim=1)
                   / (tf_raw.flatten(1).norm(dim=1) + 1e-12))
            TF_REL_DELTA.append(rel.cpu().numpy())

            F_RGB.append(F.adaptive_avg_pool2d(x, (28, 28)).flatten(1).cpu().numpy())
            F_PE.append(pe_grid.flatten(1).cpu().numpy())
            F_TF.append(tf_pe.flatten(1).cpu().numpy())
            Z_AE.append(z.flatten(1).cpu().numpy())
            LAB += [ci for _, ci, _ in batch]

            # spatial-resolution floor: what a purely spatial grid of this size holds
            for g, acc in ((g_, floors_acc[g_]) for g_ in GRIDS):
                down = F.interpolate(x, size=(g, g), mode="area")
                up = F.interpolate(down, size=(224, 224), mode="bilinear", align_corners=False)
                for k in range(x.shape[0]):
                    a = x[k].permute(1, 2, 0).cpu().numpy()
                    b = up[k].permute(1, 2, 0).cpu().numpy()
                    acc.append({"mse": float(np.mean((a - b) ** 2)),
                                "psnr": psnr(a, b), "ssim": ssim(a, b)})
            if (i // args.batch_size) % 5 == 0:
                print(f"  {min(i + args.batch_size, len(samples))}/{len(samples)}")

    cat = lambda L: np.concatenate(L).astype(np.float32)  # noqa: E731
    F_RGB, F_PE, F_TF, Z_AE = cat(F_RGB), cat(F_PE), cat(F_TF), cat(Z_AE)
    labels = np.array(LAB)
    pe_delta = cat(PE_DELTA)
    tf_rel = np.concatenate(TF_REL_DELTA)

    # ---- PE branch audit -------------------------------------------------- #
    raw_grid_std = float(F_PE.std(axis=0).mean())
    pe_offset_mean = pe_delta.mean(axis=0)          # constant component of the PE effect
    pe_offset_norm = float(np.linalg.norm(pe_offset_mean))
    pe_delta_norm = float(np.linalg.norm(pe_delta, axis=1).mean())
    f_pe_norm = float(np.linalg.norm(F_PE, axis=1).mean())
    f_tf_norm = float(np.linalg.norm(F_TF, axis=1).mean())

    pe_audit = {
        "gamma": gamma,
        "F_PE_dims": int(F_PE.shape[1]),
        "F_TF_dims": int(F_TF.shape[1]),
        "F_PE_share_of_fused_channels": round(3 / 323, 5),
        "F_PE_norm_mean": f_pe_norm,
        "F_TF_norm_mean": f_tf_norm,
        "F_PE_relative_magnitude_in_fusion": float(f_pe_norm / (f_pe_norm + f_tf_norm)),
        "pe_offset_norm_constant_component": pe_offset_norm,
        "pe_delta_norm_mean_total": pe_delta_norm,
        "pe_offset_fraction_of_delta": float(pe_offset_norm / (pe_delta_norm + 1e-12)),
        "F_PE_across_image_std_mean": raw_grid_std,
        "pe_offset_vs_image_variation": float(pe_offset_norm / (raw_grid_std * np.sqrt(F_PE.shape[1]) + 1e-12)),
        "tf_relative_change_from_PE_mean": float(tf_rel.mean()),
        "tf_relative_change_from_PE_std": float(tf_rel.std()),
    }

    # across-image variation of TF, for comparison with PE's effect on TF
    tfn = F_TF / (np.linalg.norm(F_TF, axis=1, keepdims=True) + 1e-12)
    ridx = np.random.default_rng(1).choice(len(tfn), size=min(200, len(tfn)), replace=False)
    d_between = np.linalg.norm(tfn[ridx][:, None, :] - tfn[ridx][None, :, :], axis=2)
    iu = np.triu_indices(len(ridx), 1)
    pe_audit["tf_between_image_relative_distance_mean"] = float(d_between[iu].mean())
    pe_audit["pe_effect_vs_between_image_ratio"] = float(
        pe_audit["tf_relative_change_from_PE_mean"] / (d_between[iu].mean() + 1e-12))

    # ---- representation statistics + separability ------------------------- #
    stats = [feature_stats(f, n) for f, n in
             ((F_RGB, "F_RGB (28x28 pooled pixels)"), (F_PE, "F_PE (3x7x7)"),
              (F_TF, "F_TF (320x7x7)"), (Z_AE, "Z_AE (64x7x7, random encoder)"))]
    sep = {n: separability(f, labels) for f, n in
           ((F_RGB, "F_RGB"), (F_PE, "F_PE"), (F_TF, "F_TF"), (Z_AE, "Z_AE"))}

    agg = lambda L: {k: float(np.mean([r[k] for r in L])) for k in ("mse", "psnr", "ssim")}  # noqa: E731
    floors = {}
    for g in GRIDS:
        f = agg(floors_acc[g])
        f["cell_px"] = 224 // g
        f["spatial_compression_vs_224"] = round((224 * 224) / (g * g), 1)
        f["latent_dims_at_64ch"] = 64 * g * g
        f["latent_compression_vs_input"] = round(3 * 224 * 224 / (64 * g * g), 2)
        floors[f"grid_{g}x{g}"] = f

    report = {
        "purpose": "Stage C2-2 information-preservation audit for candidate C2",
        "trained": False,
        "reconstruction_metrics_reported": False,
        "reconstruction_metrics_omitted_because":
            "the auto-encoder is randomly initialised; its decoder emits a near-constant "
            "~0.497 grey field, so MSE/PSNR/SSIM on it would measure initialisation, "
            "not architecture",
        "subset": {"split": "val", "per_class": args.per_class, "num_images": len(samples),
                   "num_classes": len(classes), "deterministic": True,
                   "selection": "sorted classes, sorted filenames, first K per class"},
        "device": device,
        "candidate": C2,
        "latent_shape": [64, grid, grid],
        "latent_dims": 64 * grid * grid,
        "input_dims": 3 * 224 * 224,
        "compression_ratio": round(3 * 224 * 224 / (64 * grid * grid), 2),
        "spatial_resolution_floor": floors,
        "pe_branch_audit": pe_audit,
        "representation_statistics": stats,
        "class_separability": sep,
        "environment": environment_info(),
    }
    with open(os.path.join(out_dir, "bottleneck_audit.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    # ---- console summary --------------------------------------------------- #
    print("\n=== spatial-resolution floor (down->up through a grid, identical images) ===")
    print(f"  {'grid':<12}{'cell':>7}{'MSE':>10}{'PSNR':>9}{'SSIM':>8}{'latent@64ch':>13}{'compr.':>9}")
    for k, v in floors.items():
        print(f"  {k:<12}{str(v['cell_px'])+'px':>7}{v['mse']:>10.5f}{v['psnr']:>9.2f}"
              f"{v['ssim']:>8.4f}{v['latent_dims_at_64ch']:>13,}{v['latent_compression_vs_input']:>8.1f}x")

    print("\n=== representation statistics ===")
    print(f"{'representation':<32}{'dims':>7}{'norm':>10}{'across/within':>15}{'cos(img_i,img_j)':>18}")
    for s in stats:
        print(f"{s['name']:<32}{s['dims']:>7}{s['feature_norm_mean']:>10.3f}"
              f"{s['across_over_within']:>15.4f}{s['cosine_across_images_mean']:>18.4f}")

    print("\n=== class separability (LOO-style CV, cosine) ===")
    print(f"{'representation':<12}{'1-NN acc':>10}{'silhouette':>13}   (chance = "
          f"{1/len(classes):.4f})")
    for n, s in sep.items():
        print(f"{n:<12}{s.get('knn1_cv_accuracy', float('nan')):>10.4f}"
              f"{s.get('silhouette', float('nan')):>13.4f}")

    print("\n=== PE branch audit ===")
    p = pe_audit
    print(f"  F_PE is {p['F_PE_share_of_fused_channels']*100:.2f}% of fused channels, "
          f"{p['F_PE_relative_magnitude_in_fusion']*100:.1f}% of fused magnitude")
    print(f"  PE offset (constant component)          : {p['pe_offset_norm_constant_component']:.5f}")
    print(f"  PE total delta at fusion input          : {p['pe_delta_norm_mean_total']:.5f}")
    print(f"  -> constant fraction of PE's effect     : {p['pe_offset_fraction_of_delta']*100:.1f}%")
    print(f"  PE-induced relative change in F_TF      : {p['tf_relative_change_from_PE_mean']:.5f}"
          f" +/- {p['tf_relative_change_from_PE_std']:.5f}")
    print(f"  between-image relative distance in F_TF : {p['tf_between_image_relative_distance_mean']:.5f}")
    print(f"  -> PE effect / between-image variation  : {p['pe_effect_vs_between_image_ratio']:.4f}")

    if not args.no_figures:
        make_figures(samples, classes, out_dir, device, grid)

    print(f"\nwrote {out_dir}/bottleneck_audit.json")
    return 0


# ------------------------------------------------------------------ figures -- #

REPRESENTATIVE = {
    "healthy": ["Apple___healthy", "Tomato___healthy"],
    "large lesions": ["Tomato___Late_blight", "Grape___Black_rot"],
    "small localized spots": ["Apple___Cedar_apple_rust", "Tomato___Septoria_leaf_spot"],
    "texture-dominated": ["Squash___Powdery_mildew", "Cherry___Powdery_mildew"],
    "visually similar pair": ["Tomato___Early_blight", "Tomato___Bacterial_spot"],
}


def make_figures(samples, classes, out_dir, device, grid) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_class = {}
    for p, ci, c in samples:
        by_class.setdefault(c, []).append(p)

    rows = []
    for group, names in REPRESENTATIVE.items():
        for c in names:
            if c in by_class:
                rows.append((group, c, by_class[c][0]))

    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 5, figsize=(13.5, 2.5 * len(rows)), dpi=130)
    if len(rows) == 1:
        axes = axes[None, :]
    for r, (group, cname, path) in enumerate(rows):
        img = np.asarray(Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC),
                         dtype=np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1)[None]
        panels = [("original", img)]
        for g in (28, 14, 7):
            up = F.interpolate(F.interpolate(x, size=(g, g), mode="area"),
                               size=(224, 224), mode="bilinear", align_corners=False)
            panels.append((f"{g}x{g} spatial floor", up[0].permute(1, 2, 0).numpy()))
        err = np.abs(img - panels[-1][1]).mean(axis=2)
        panels.append(("|error| at 7x7", err))

        for k, (title, arr) in enumerate(panels):
            ax = axes[r, k]
            ax.imshow(arr, cmap="inferno" if arr.ndim == 2 else None,
                      vmin=0, vmax=(err.max() if arr.ndim == 2 else 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
            if k == 0:
                ax.set_ylabel(f"{group}\n{cname[:22]}", fontsize=6.5, rotation=0,
                              ha="right", va="center", labelpad=42)
    fig.suptitle("Spatial floor at each candidate grid, identical images "
                 "(3-channel floor -- the 64-channel latent can carry more)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(out_dir, "spatial_floor_grid.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
