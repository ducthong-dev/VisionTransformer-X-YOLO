#!/usr/bin/env python
"""t-SNE / UMAP of clean vs corrupted representations, before and after the AE.

Consumes the .npz written by analyze_latent_stability.py --save-embeddings, so
the picture and the numbers always come from the same extraction.

    python scripts/analyze_latent_stability.py --run <run> --corruption pepper/030 --save-embeddings
    python scripts/plot_latents.py --run <run> --corruption pepper/030

The figure is illustrative. The quantitative drift statistics, not the plot, are
what support the robustness claim.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.seeding import seed_everything  # noqa: E402


def project(x: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=seed).fit_transform(x)
        except ImportError:
            print("umap-learn not installed; falling back to t-SNE")
    from sklearn.manifold import TSNE

    perplexity = float(min(30, max(5, (len(x) - 1) / 3)))
    return TSNE(n_components=2, random_state=seed, init="pca",
                perplexity=perplexity).fit_transform(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--corruption", default="pepper/030")
    ap.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    ap.add_argument("--max-points", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    tag = args.corruption.replace("/", "_")
    npz_path = os.path.join(args.run, "latent", f"embeddings_{tag}.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(
            f"no embeddings at {npz_path}\n"
            f"run: python scripts/analyze_latent_stability.py --run {args.run} "
            f"--corruption {args.corruption} --save-embeddings"
        )

    z = np.load(npz_path)
    labels = z["labels"]
    n = min(args.max_points, len(labels))
    idx = np.random.default_rng(args.seed).choice(len(labels), n, replace=False)
    labels = labels[idx]

    panels = [("before AE", z["pre_clean"][idx], z["pre_corrupt"][idx])]
    if "ae_clean" in z:
        panels.append(("AE latent", z["ae_clean"][idx], z["ae_corrupt"][idx]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6.2), dpi=150)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, clean, corrupt) in zip(axes, panels):
        # project clean and corrupted jointly so the two are in the same space
        joint = project(np.concatenate([clean, corrupt]), args.method, args.seed)
        c2, x2 = joint[: len(clean)], joint[len(clean):]
        ax.scatter(c2[:, 0], c2[:, 1], c=labels, cmap="tab20", s=14, alpha=0.85,
                   marker="o", linewidths=0)
        ax.scatter(x2[:, 0], x2[:, 1], c=labels, cmap="tab20", s=22, alpha=0.55,
                   marker="x", linewidths=0.8)
        drift = float(np.linalg.norm(c2 - x2, axis=1).mean())
        ax.set_title(f"{title}\nclean = circle, corrupted = cross\n"
                     f"mean 2-D displacement {drift:.2f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{os.path.basename(args.run)} - {args.corruption} - {args.method.upper()}",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.run, "latent", f"{args.method}_{tag}.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
