"""Datasets and the frozen training/evaluation transform protocol.

Design rules that make the ablation fair:

* Images are delivered to the model as float in [0, 1]; ALL normalisation that a
  component needs (e.g. ViT's (x-0.5)/0.5) happens inside that component. So
  every arm sees byte-identical pixels.
* Geometric/photometric training augmentation is applied to the raw image BEFORE
  the front-end, so PE, TF and AE all see a consistent view.
* The class list is derived once from the training split and reused everywhere,
  so a corrupted directory that happens to be missing a class cannot silently
  shift label indices.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".JPG", ".JPEG", ".PNG")


def list_classes(root: str) -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def list_samples(root: str, classes: list[str]) -> list[tuple[str, int]]:
    idx = {c: i for i, c in enumerate(classes)}
    out: list[tuple[str, int]] = []
    for c in classes:
        d = os.path.join(root, c)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(IMG_EXT):
                out.append((os.path.join(d, f), idx[c]))
    return out


@dataclass
class TrainAugConfig:
    """Frozen training augmentation. Identical for every arm."""

    hflip: float = 0.5
    randaugment: bool = True
    randaugment_n: int = 2
    randaugment_m: int = 9
    erasing: float = 0.4


class LeafDataset(Dataset):
    """ImageFolder-style dataset returning ([3,H,W] float in [0,1], label, path)."""

    def __init__(
        self,
        root: str,
        classes: list[str],
        img_size: int = 224,
        train: bool = False,
        aug: TrainAugConfig | None = None,
        return_path: bool = False,
    ) -> None:
        self.root = root
        self.classes = classes
        self.img_size = img_size
        self.train = train
        self.return_path = return_path
        self.samples = list_samples(root, classes)
        if not self.samples:
            raise RuntimeError(f"no images found under {root}")

        aug = aug or TrainAugConfig()
        ops: list = [transforms.Resize((img_size, img_size))]
        if train:
            if aug.hflip > 0:
                ops.append(transforms.RandomHorizontalFlip(p=aug.hflip))
            if aug.randaugment:
                ops.append(
                    transforms.RandAugment(num_ops=aug.randaugment_n, magnitude=aug.randaugment_m)
                )
        ops.append(transforms.ToTensor())          # -> [0, 1], no normalisation
        if train and aug.erasing > 0:
            ops.append(transforms.RandomErasing(p=aug.erasing, value=0))
        self.tf = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        if self.return_path:
            return x, label, path
        return x, label


class PairedCorruptionDataset(Dataset):
    """Yields (clean, corrupted, label, rel_path) for the latent-stability analysis.

    Both roots must contain the same relative paths; the corrupted root is one of
    the frozen directories written by scripts/generate_corruptions.py.
    """

    def __init__(self, clean_root: str, corrupt_root: str, classes: list[str], img_size: int = 224):
        self.clean_root = clean_root
        self.corrupt_root = corrupt_root
        self.classes = classes
        self.tf = transforms.Compose(
            [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
        )
        self.items: list[tuple[str, str, int]] = []
        idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            cd = os.path.join(clean_root, c)
            if not os.path.isdir(cd):
                continue
            for f in sorted(os.listdir(cd)):
                if not f.endswith(IMG_EXT):
                    continue
                stem = os.path.splitext(f)[0]
                cand = os.path.join(corrupt_root, c, stem + ".png")
                if os.path.exists(cand):
                    self.items.append((os.path.join(cd, f), cand, idx[c]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        cp, xp, label = self.items[i]
        clean = self.tf(Image.open(cp).convert("RGB"))
        corrupt = self.tf(Image.open(xp).convert("RGB"))
        return clean, corrupt, label, os.path.relpath(cp, self.clean_root)


# ------------------------------------------------------------------------- #
# Training-time corruption, used by (a) the AE denoising objective and
# (b) arm A4, the corruption-augmented-training control.
#
# This is deliberately a DIFFERENT process from the frozen test benchmark: it
# draws severities continuously at random per batch, whereas the benchmark uses
# fixed discrete severities on fixed files. Keeping them distinct is what stops
# A4 from being trained on its own test set.
# ------------------------------------------------------------------------- #


def training_corruption(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Mild random Gaussian + impulse noise on a [B,3,H,W] batch in [0,1]."""
    b = x.shape[0]
    dev = x.device
    out = x.clone()

    sigma = torch.rand(b, 1, 1, 1, device=dev, generator=generator) * 0.15
    out = out + torch.randn(x.shape, device=dev, generator=generator) * sigma

    ratio = torch.rand(b, 1, 1, 1, device=dev, generator=generator) * 0.10
    u = torch.rand(x.shape[0], 1, *x.shape[2:], device=dev, generator=generator)
    salt = torch.rand(x.shape[0], 1, *x.shape[2:], device=dev, generator=generator) < 0.5
    hit = u < ratio
    out = torch.where(hit & salt, torch.ones_like(out), out)
    out = torch.where(hit & ~salt, torch.zeros_like(out), out)

    return out.clamp(0.0, 1.0)


def dataset_fingerprint(root: str, classes: list[str]) -> dict:
    """Cheap structural hash of a split, recorded in every result JSON."""
    import hashlib

    h = hashlib.sha256()
    total = 0
    per_class = {}
    for c in classes:
        d = os.path.join(root, c)
        files = sorted(f for f in os.listdir(d) if f.endswith(IMG_EXT)) if os.path.isdir(d) else []
        per_class[c] = len(files)
        total += len(files)
        h.update(c.encode())
        for f in files:
            h.update(f.encode())
    return {"root": root, "num_classes": len(classes), "num_images": total,
            "per_class": per_class, "listing_sha256": h.hexdigest()}
