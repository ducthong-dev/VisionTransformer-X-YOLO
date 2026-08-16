"""Lightweight global-context encoders for the v2 architecture evaluation.

EXPLORATORY -- added for docs/ARCHITECTURE_V2_BENCHMARK.md. Nothing in the frozen
`revision-protocol-v1` matrix uses this module; the frozen TF branch remains
`TransformerFeatureRGB` (HuggingFace ViT-B/16) and is untouched.

Same output contract as `TransformerFeatureRGB` so it is a drop-in substitute:

    forward(x)          -> [B, out_channels, H, W] in [0, 1]   (image-space TF-RGB)
    forward_features(x) -> [B, C_backbone, h, w]               (native grid, for
                                                                feature-space fusion)

`out_channels`, the 1x1 projection, the BatchNorm, the sigmoid and the bilinear
upsample are all identical to the ViT-B/16 branch, so swapping the encoder
changes the encoder and nothing else about the fusion interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet statistics: what every timm ImageNet checkpoint expects. The ViT-B/16
# branch uses mean=std=0.5 because that is what its in21k checkpoint expects;
# the normalisation belongs to the checkpoint, not to the architecture.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TimmGlobalContextRGB(nn.Module):
    """A timm backbone used as the TF-RGB branch.

    Candidates evaluated (all ImageNet-pretrained, all carrying genuine
    global-context blocks):

        mobilevit_xxs     0.95 M   spatial self-attention on unfolded patches
        efficientvit_b0   0.68 M   multi-scale linear attention
        edgenext_xx_small 1.16 M   split depth-wise transpose attention
    """

    def __init__(
        self,
        model_name: str = "mobilevit_xxs",
        out_channels: int = 3,
        freeze: bool = True,
        pretrained: bool = True,
        upsample_mode: str = "bilinear",
        out_index: int | None = None,
    ) -> None:
        """`out_index` selects which backbone stage supplies the representation.

        None -> the final stage (stride 32, 7x7 at 224 input). Passing an earlier
        index both raises the spatial resolution AND truncates the backbone, so
        the network beyond that stage is never built or executed. For
        mobilevit_xxs at 224:

            index 2 -> 48 ch, 28x28, stride 8    149,520 params, 0.3210 GFLOPs
            index 3 -> 64 ch, 14x14, stride 16   491,344 params, 0.4598 GFLOPs
            index 4 -> 320 ch,  7x7, stride 32   951,024 params, 0.5137 GFLOPs

        All three of those stages terminate in a MobileVitBlock, so each carries
        genuine global context -- stages 0 and 1 are purely convolutional
        (BottleneckBlock only) and must not be used as the TF branch.
        """
        super().__init__()
        import timm

        kwargs = {} if out_index is None else {"out_indices": (out_index,)}
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, features_only=True, **kwargs)
        self.model_name = model_name
        self.out_index = out_index
        self.frozen = bool(freeze)
        self.upsample_mode = upsample_mode

        info = self.backbone.feature_info
        self.backbone_channels = info.channels()[-1]
        self.backbone_reduction = info.reduction()[-1]

        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.project = nn.Conv2d(self.backbone_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """-> [B, C_backbone, h, w] at the backbone's native stride."""
        normed = (x - self.mean) / self.std
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            return self.backbone(normed)[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """-> [B, out_channels, H, W] in [0, 1], matching TransformerFeatureRGB."""
        h, w = x.shape[-2:]
        fmap = torch.sigmoid(self.norm(self.project(self.forward_features(x))))
        return F.interpolate(fmap, size=(h, w), mode=self.upsample_mode, align_corners=False)

    def describe(self) -> dict:
        return {
            "model_name": self.model_name,
            "out_index": self.out_index,
            "library": "timm (features_only)",
            "pretrained": True,
            "frozen": self.frozen,
            "backbone_channels": self.backbone_channels,
            "backbone_reduction": self.backbone_reduction,
            "output_representation": "last feature stage, 1x1-projected and upsampled",
            "input_normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        }
