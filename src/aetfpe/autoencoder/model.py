"""Stacked sparse denoising auto-encoder used as the AE-TFPE fusion operator.

RECONSTRUCTED MODULE. No auto-encoder of any kind survives in any recovered
artifact (docs/ARCHITECTURE_RECOVERY.md, goals 5-6). The manuscript calls it
"stacked" (title), "sparse" (S3.3) and "a denoising latent regularizer" (S1) --
three different things. This implementation resolves the ambiguity in the only
direction that supports the paper's actual claim of noise resilience:

  * corrupted input, clean reconstruction target  -> denoising
  * KL sparsity penalty on latent channel means   -> sparse
  * three encoder stages / three decoder stages   -> stacked

so the correct term, and the one the revised manuscript must use, is
**stacked sparse denoising auto-encoder**.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def _up_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class StackedSparseDenoisingAE(nn.Module):
    """Conv auto-encoder mapping a fused multi-channel map back to an RGB image.

    Input   : [B, in_channels, 224, 224]   (fused PE-RGB + TF-RGB, or plain RGB)
    Latent  : [B, latent_channels, 28, 28] with sigmoid activation, so channel
              means are valid Bernoulli parameters for the KL sparsity term
    Output  : [B, 3, 224, 224] in [0, 1]   (reconstruction, consumed by YOLO)

    Three stride-2 encoder stages take 224 -> 112 -> 56 -> 28; three transposed
    stages take it back. `latent_channels` x 28 x 28 is the latent dimension.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        widths: tuple[int, ...] = (32, 64),
        latent_channels: int = 128,
    ) -> None:
        super().__init__()
        w1, w2 = widths
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.encoder = nn.Sequential(
            _block(in_channels, w1, stride=2),      # 224 -> 112
            _block(w1, w2, stride=2),               # 112 -> 56
            nn.Conv2d(w2, latent_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.Sigmoid(),                           # 56 -> 28, latent in (0, 1)
        )
        self.decoder = nn.Sequential(
            _up_block(latent_channels, w2),         # 28 -> 56
            _up_block(w2, w1),                      # 56 -> 112
            nn.ConvTranspose2d(w1, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),                           # 112 -> 224, output in (0, 1)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon

    def latent_dim(self, spatial: int = 28) -> int:
        return self.latent_channels * spatial * spatial

    def describe(self) -> dict:
        return {
            "type": "stacked sparse denoising auto-encoder",
            "in_channels": self.in_channels,
            "encoder": "3 x [Conv3x3 s2 - BN - ReLU/Sigmoid], 224->112->56->28",
            "decoder": "3 x [ConvT4x4 s2 - BN - ReLU/Sigmoid], 28->56->112->224",
            "latent_shape": [self.latent_channels, 28, 28],
            "latent_dim": self.latent_dim(),
            "latent_activation": "sigmoid",
            "output_activation": "sigmoid",
        }


class SlimFeatureSpaceAE(nn.Module):
    """Feature-space variant of the stacked sparse denoising auto-encoder.

    EXPLORATORY -- added for docs/ARCHITECTURE_V2_BENCHMARK.md. The frozen
    `revision-protocol-v1` matrix uses StackedSparseDenoisingAE above, unchanged.

    Motivation: manuscript Section 5.1 states the auto-encoder "operates on
    intermediate feature representations rather than raw images", and Section 5.3
    that fusion works on "fixed-dimensional latent features rather than directly
    on image pixels". The image-space AE contradicts both -- and that contradiction
    is what makes it cost 5x the classifier's FLOPs on only 259 K parameters,
    because it runs three convolutional stages at full 224x224.

    This variant encodes at the transformer's native grid (e.g. 7x7), where the
    fused representation already lives, and decodes back to an image so the
    classifier stays completely unmodified.

    Input   : [B, in_channels, h, w]        fused map at the encoder's grid
    Latent  : [B, latent_channels, h, w]    sigmoid, so channel means remain valid
                                            Bernoulli parameters for the KL term
    Output  : [B, 3, out_size, out_size] in [0, 1]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        latent_channels: int = 64,
        decoder_widths: tuple[int, ...] = (48, 32, 16, 8),
        grid: int = 7,
        out_size: int = 224,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.grid = grid
        self.out_size = out_size

        n_up = 0
        s = grid
        while s < out_size:
            s *= 2
            n_up += 1
        if s != out_size:
            raise ValueError(f"grid {grid} cannot reach {out_size} by doubling")
        if n_up != len(decoder_widths) + 1:
            raise ValueError(
                f"grid {grid}->{out_size} needs {n_up} upsampling stages, but "
                f"decoder_widths has {len(decoder_widths)} entries (expected {n_up - 1})"
            )

        # Encoder: a 1x1 bottleneck at the grid. No spatial reduction -- the
        # backbone has already done it, which is the entire point of this variant.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.Sigmoid(),
        )

        layers: list[nn.Module] = []
        cin = latent_channels
        for cout in decoder_widths:
            layers.append(_up_block(cin, cout))
            cin = cout
        layers.append(nn.ConvTranspose2d(cin, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, return_latent: bool = False):
        z = self.encode(x)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon

    def latent_dim(self) -> int:
        return self.latent_channels * self.grid * self.grid

    def describe(self) -> dict:
        return {
            "type": "slim feature-space sparse denoising auto-encoder",
            "in_channels": self.in_channels,
            "encoder": f"Conv1x1 -> BN -> Sigmoid at {self.grid}x{self.grid} (no spatial reduction)",
            "decoder": f"{len(self.decoder) // 2} x [ConvT4x4 s2], {self.grid} -> {self.out_size}",
            "latent_shape": [self.latent_channels, self.grid, self.grid],
            "latent_dim": self.latent_dim(),
            "latent_activation": "sigmoid",
            "output_activation": "sigmoid",
            "operates_on": "intermediate feature representation (consistent with manuscript 5.1/5.3)",
        }
