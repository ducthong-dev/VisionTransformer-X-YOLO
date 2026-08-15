"""Classifier backbones, all behind one constructor so every arm is comparable.

YOLOv8n-cls is the model the manuscript uses. It is built here as a plain
`nn.Module` (`ultralytics.nn.tasks.ClassificationModel`) rather than through the
Ultralytics trainer, so that every ablation arm shares one training loop and one
protocol. That is a deliberate reconstruction decision; see
docs/ARCHITECTURE_RECOVERY.md, "Training protocol".

ResNet-50, EfficientNet-B0 and ViT-B/16 are provided for the Tier E fair-baseline
comparison and go through the identical loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _transfer_matching(dst: nn.Module, src_state: dict) -> tuple[int, int]:
    """Copy every tensor whose name and shape match. Returns (copied, total)."""
    dst_state = dst.state_dict()
    copied = 0
    for k, v in src_state.items():
        if k in dst_state and dst_state[k].shape == v.shape:
            dst_state[k] = v.clone()
            copied += 1
    dst.load_state_dict(dst_state, strict=True)
    return copied, len(dst_state)


def adapt_stem(model: nn.Module, in_channels: int) -> nn.Module:
    """Widen the first Conv2d of `model` to accept `in_channels` inputs.

    Only used by the plain-concatenation fusion arm. Existing weights are tiled
    and rescaled so the effective response to a duplicated input is unchanged,
    which avoids handing that arm a worse initialisation than the others.
    """
    if in_channels == 3:
        return model
    first = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first = module
            break
    if first is None or first.in_channels == in_channels:
        return model

    old_w = first.weight.data                       # [out, 3, k, k]
    reps = in_channels // old_w.shape[1]
    new_w = old_w.repeat(1, reps, 1, 1) / reps
    first.in_channels = in_channels
    first.weight = nn.Parameter(new_w.contiguous())
    return model


def build_classifier(
    name: str = "yolov8n-cls",
    num_classes: int = 39,
    pretrained: bool = True,
    in_channels: int = 3,
) -> nn.Module:
    name = name.lower()

    if name.startswith("yolo"):
        from ultralytics import YOLO
        from ultralytics.nn.tasks import ClassificationModel

        model = ClassificationModel(cfg=f"{name}.yaml", nc=num_classes, verbose=False)
        if pretrained:
            src = YOLO(f"{name}.pt").model.state_dict()
            copied, total = _transfer_matching(model, src)
            model._pretrained_transfer = f"{copied}/{total}"
        else:
            model._pretrained_transfer = "0/0 (scratch)"
        return adapt_stem(model, in_channels)

    import torchvision.models as tvm

    if name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name in ("efficientnet_b0", "efficientnet-b0"):
        m = tvm.efficientnet_b0(
            weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name in ("vit_b_16", "vit-b-16", "vit_b16"):
        m = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise KeyError(f"unknown classifier {name!r}")

    m._pretrained_transfer = "torchvision weights" if pretrained else "scratch"
    return adapt_stem(m, in_channels)


def classifier_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Ultralytics ClassificationModel returns logits directly; so do the rest."""
    out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out
