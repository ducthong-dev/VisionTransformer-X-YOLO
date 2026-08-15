"""TF-RGB: transformer-derived features, rendered back into image space.

RECOVERED + RECONSTRUCTED.

RECOVERED (HIGH confidence): the historical code named the checkpoint
`google/vit-base-patch16-224-in21k` but only ever called `ViTImageProcessor`,
i.e. the *preprocessor*. No transformer forward pass was executed
(feature_extractor_folder.py:3,10,13-16; verified pixel-exactly against the
surviving processed dataset). The checkpoint identity is therefore the only
recovered fact; everything below is reconstruction.

RECONSTRUCTED: an actual frozen ViT-B/16 forward pass over PE-RGB, with the
patch tokens reshaped back to a 14x14 grid, projected to 3 channels and upsampled
to image resolution. This keeps the whole front-end in image space, which is what
the manuscript's Fig. 2 depicts and what lets an unmodified YOLOv8n-cls consume
the result.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ViT normalisation for the in21k checkpoints: mean = std = 0.5
VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD = (0.5, 0.5, 0.5)


class TransformerFeatureRGB(nn.Module):
    """PE-RGB -> ViT-B/16 -> patch-token map -> 3-channel image-space feature.

    Input  : [B, 3, 224, 224] float in [0, 1]
    ViT    : last_hidden_state [B, 197, 768]  (1 CLS + 196 patch tokens)
    Grid   : [B, 768, 14, 14]
    Project: [B, out_channels, 14, 14]        (1x1 conv, trainable)
    Output : [B, out_channels, 224, 224] float in [0, 1]

    The ViT itself is frozen by default: the manuscript never claims to fine-tune
    it, and freezing keeps TF-RGB deterministic given the image, which lets the
    map be precomputed once for the whole dataset (see docs).
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        out_channels: int = 3,
        freeze: bool = True,
        pretrained: bool = True,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        from transformers import ViTConfig, ViTModel

        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        else:
            self.vit = ViTModel(ViTConfig(), add_pooling_layer=False)

        cfg = self.vit.config
        self.hidden_size = cfg.hidden_size            # 768
        self.patch_size = cfg.patch_size              # 16
        self.num_layers = cfg.num_hidden_layers       # 12
        self.num_heads = cfg.num_attention_heads      # 12
        self.mlp_dim = cfg.intermediate_size          # 3072
        self.model_name = model_name
        self.frozen = bool(freeze)
        self.upsample_mode = upsample_mode

        if freeze:
            self.vit.eval()
            for p in self.vit.parameters():
                p.requires_grad_(False)

        self.project = nn.Conv2d(self.hidden_size, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)

        self.register_buffer("vit_mean", torch.tensor(VIT_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("vit_std", torch.tensor(VIT_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):  # keep a frozen ViT in eval mode always
        super().train(mode)
        if self.frozen:
            self.vit.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        normed = (x - self.vit_mean) / self.vit_std

        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.vit(pixel_values=normed).last_hidden_state    # [B, 197, 768]

        tokens = out[:, 1:, :]                                       # drop CLS -> [B, 196, 768]
        grid = int(tokens.shape[1] ** 0.5)
        if grid * grid != tokens.shape[1]:
            raise ValueError(f"non-square token grid: {tokens.shape[1]} tokens")
        fmap = tokens.transpose(1, 2).reshape(b, self.hidden_size, grid, grid)

        fmap = self.norm(self.project(fmap))
        fmap = torch.sigmoid(fmap)                                   # into [0, 1], image-like
        return F.interpolate(fmap, size=(h, w), mode=self.upsample_mode, align_corners=False)

    def describe(self) -> dict:
        return {
            "model_name": self.model_name,
            "library": "transformers.ViTModel",
            "pretrained": True,
            "frozen": self.frozen,
            "patch_size": self.patch_size,
            "embedding_dim": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "output_representation": "patch tokens (CLS dropped), 1x1-projected and upsampled",
            "input_normalization": {"mean": VIT_MEAN, "std": VIT_STD},
        }
