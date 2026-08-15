#!/usr/bin/env python
"""Train one arm under the frozen protocol.

    python scripts/train.py --config configs/aetfpe_full.yaml
    python scripts/train.py --config configs/baseline_rgb.yaml --epochs 1 --limit-per-class 4

One loop serves every arm, so "identical experimental conditions" is enforced by
construction rather than asserted. When the arm has an auto-encoder, the AE loss
is optimised jointly with the classification loss:

    L = CE(f(x_hat), y) + w * [ MSE(x_hat, x_clean) + beta * KL(rho || rho_hat) ]

with the AE input corrupted and the reconstruction target clean, which is what
makes it a denoising objective.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.autoencoder.losses import ae_loss  # noqa: E402
from aetfpe.config import (  # noqa: E402
    build_protocol, environment_info, load_experiment, pick_device, resolve_roots,
    save_run_provenance,
)
from aetfpe.data import (  # noqa: E402
    LeafDataset, dataset_fingerprint, list_classes, training_corruption,
)
from aetfpe.metrics import summarize  # noqa: E402
from aetfpe.models import build_model  # noqa: E402
from aetfpe.models.classifier import classifier_forward  # noqa: E402
from aetfpe.seeding import seed_everything  # noqa: E402


def _coerce(text: str):
    """Parse an override value as YAML so bools, ints, floats and strings all work."""
    import yaml as _yaml

    return _yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply `dotted.key=value` overrides in place, creating intermediate dicts.

    Overrides land in the config *before* the protocol is built, so they are
    captured in the run's saved config.yaml and are therefore part of the
    provenance record rather than an invisible command-line flag.
    """
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--override expects DOTTED.KEY=VALUE, got {item!r}")
        path, _, raw = item.partition("=")
        node = cfg
        parts = path.strip().split(".")
        for key in parts[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise SystemExit(f"--override path {path!r} traverses a non-mapping")
        node[parts[-1]] = _coerce(raw)
    if overrides:
        cfg.setdefault("_overrides", []).extend(overrides)


def subset(ds: LeafDataset, per_class: int | None):
    if not per_class:
        return ds
    seen: dict[int, int] = {}
    keep = []
    for i, (_, lab) in enumerate(ds.samples):
        if seen.get(lab, 0) < per_class:
            keep.append(ds.samples[i])
            seen[lab] = seen.get(lab, 0) + 1
    ds.samples = keep
    return ds


def run_epoch(model, loader, device, protocol, cfg, optimizer=None, scheduler=None,
              ae_only: bool = False):
    """One pass. `ae_only` runs the stacked-AE warm-up: reconstruction loss only,
    no classification gradient, so the AE learns to pass an image through before
    the classifier is asked to read one."""
    train = optimizer is not None
    model.train(train)
    ce = nn.CrossEntropyLoss(label_smoothing=protocol.label_smoothing)
    use_ae = model.cfg.use_ae
    denoise = model.cfg.ae_denoising
    aug_corrupt = bool((cfg.get("train") or {}).get("corruption_augmentation", False))

    tot = correct = 0
    loss_sum = 0.0
    comps = {"ae_recon": 0.0, "ae_kl": 0.0}

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x_clean = x

        if train and aug_corrupt:
            x = training_corruption(x)
        if train and use_ae and denoise:
            x = training_corruption(x)      # AE sees a corrupted view, target stays clean

        with torch.set_grad_enabled(train):
            out, parts = model.frontend(x, return_parts=True)
            logits = classifier_forward(model.classifier, out)
            ce_loss = ce(logits, y)
            loss = torch.zeros((), device=device) if ae_only else ce_loss

            if use_ae and parts["latent"] is not None:
                a_loss, a_comp = ae_loss(
                    out, x_clean, parts["latent"],
                    beta=protocol.ae_beta, rho=protocol.ae_rho, sparse=model.cfg.ae_sparse,
                )
                loss = loss + protocol.ae_loss_weight * a_loss
                comps["ae_recon"] += a_comp["ae_recon"] * y.size(0)
                comps["ae_kl"] += a_comp["ae_kl"] * y.size(0)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        loss_sum += float(loss.detach()) * y.size(0)
        correct += int((logits.argmax(1) == y).sum())
        tot += y.size(0)

    stats = {"loss": loss_sum / max(tot, 1), "top1": correct / max(tot, 1), "n": tot}
    if use_ae and tot:
        stats["ae_recon"] = comps["ae_recon"] / tot
        stats["ae_kl"] = comps["ae_kl"] / tot
    return stats


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_logits.append(logits.float().cpu().numpy())
        all_y.append(y.numpy())
    return summarize(np.concatenate(all_logits), np.concatenate(all_y), classes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None, help="override, for smoke tests")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="cap images per class on BOTH train and val (smoke tests)")
    ap.add_argument("--limit-train-per-class", type=int, default=None,
                    help="cap the TRAINING split only, leaving validation at full size. "
                         "Use for hyperparameter-freeze runs, where the decision must be "
                         "made on the complete validation split.")
    ap.add_argument("--limit-val-per-class", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--override", action="append", default=[], metavar="DOTTED.KEY=VALUE",
                    help="override a config value, e.g. --override protocol.ae_loss_weight=1. "
                         "Repeatable. Recorded verbatim in the run's config.yaml.")
    args = ap.parse_args()

    cfg = load_experiment(args.config)
    apply_overrides(cfg, args.override)
    protocol = build_protocol(cfg)
    if args.epochs is not None:
        protocol.epochs = args.epochs
    if args.batch_size is not None:
        protocol.batch_size = args.batch_size
    if args.num_workers is not None:
        protocol.num_workers = args.num_workers

    name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = args.out or os.path.join(
        resolve_roots()["OUTPUT_ROOT"], cfg.get("group", "misc"), name)
    os.makedirs(out_dir, exist_ok=True)

    seed_everything(protocol.seed, protocol.deterministic)
    device = pick_device(args.device)

    d = cfg["data"]
    train_root = os.path.join(d["root"], d["train_split"])
    val_root = os.path.join(d["root"], d["val_split"])
    classes = list_classes(train_root)
    cfg.setdefault("model", {})["num_classes"] = len(classes)

    limit_tr = args.limit_train_per_class if args.limit_train_per_class is not None else args.limit_per_class
    limit_va = args.limit_val_per_class if args.limit_val_per_class is not None else args.limit_per_class
    tr = subset(LeafDataset(train_root, classes, protocol.img_size, train=True), limit_tr)
    va = subset(LeafDataset(val_root, classes, protocol.img_size, train=False), limit_va)

    tl = DataLoader(tr, batch_size=protocol.batch_size, shuffle=True,
                    num_workers=protocol.num_workers, drop_last=False)
    vl = DataLoader(va, batch_size=protocol.batch_size, shuffle=False,
                    num_workers=protocol.num_workers)

    model = build_model(cfg["model"]).to(device)
    desc = model.describe()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=protocol.lr, weight_decay=protocol.weight_decay,
                                  betas=tuple(protocol.betas))
    steps = max(len(tl) * protocol.epochs, 1)
    warm = max(int(len(tl) * protocol.warmup_epochs), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warm if s < warm
        else max(0.0, 0.5 * (1 + np.cos(np.pi * (s - warm) / max(steps - warm, 1)))),
    )

    save_run_provenance(out_dir, cfg, protocol, extra={
        "model": desc,
        "classes": classes,
        "train_fingerprint": dataset_fingerprint(train_root, classes),
        "val_fingerprint": dataset_fingerprint(val_root, classes),
        "device": device,
    })

    print(f"[{name}] device={device} classes={len(classes)} "
          f"train={len(tr)} val={len(va)} epochs={protocol.epochs}")
    print(f"[{name}] frontend={json.dumps({k: desc[k] for k in ('use_pe','use_tf','use_ae','fusion','legacy_lut','photometric')})}")
    print(f"[{name}] classifier_in_channels={desc['classifier_in_channels']} "
          f"stem_modified={desc['classifier_stem_modified']} transfer={desc['pretrained_transfer']}")

    history, best = [], -1.0
    ckpt = os.path.join(out_dir, "checkpoint.pt")
    t_start = time.time()

    warmup_epochs = protocol.ae_warmup_epochs if model.cfg.use_ae else 0
    if warmup_epochs:
        print(f"[{name}] stacked-AE warm-up: {warmup_epochs} reconstruction-only epochs")

    for ep in range(protocol.epochs):
        t0 = time.time()
        ae_only = ep < warmup_epochs
        tr_stats = run_epoch(model, tl, device, protocol, cfg, optimizer, scheduler, ae_only=ae_only)
        va_stats = run_epoch(model, vl, device, protocol, cfg)
        row = {"epoch": ep + 1, "seconds": round(time.time() - t0, 2),
               "lr": optimizer.param_groups[0]["lr"], "stage": "ae_warmup" if ae_only else "joint",
               **{f"train_{k}": v for k, v in tr_stats.items()},
               **{f"val_{k}": v for k, v in va_stats.items()}}
        history.append(row)
        print(f"  ep{ep+1:>3}/{protocol.epochs} [{row['stage']}] train_loss={tr_stats['loss']:.4f} "
              f"train_top1={tr_stats['top1']:.4f}  val_top1={va_stats['top1']:.4f}  "
              f"({row['seconds']}s)")

        # never select a checkpoint from the reconstruction-only warm-up
        if not ae_only and va_stats["top1"] > best:
            best = va_stats["top1"]
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "epoch": ep + 1, "val_top1": best, "classes": classes}, ckpt)

    import csv as _csv
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    summary = {
        "name": name, "group": cfg.get("group"), "config": args.config,
        "best_val_top1": best, "epochs": protocol.epochs,
        "train_seconds": round(time.time() - t_start, 1),
        "protocol": asdict(protocol), "model": desc,
        "environment": environment_info(), "device": device,
        "checkpoint": ckpt,
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"[{name}] best val top-1 = {best:.4f}  ->  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
