"""PE-RGB: positional-encoding-enhanced RGB.

RECONSTRUCTED MODULE. No implementation of manuscript Eq. (1) survives in any
recovered artifact (see docs/ARCHITECTURE_RECOVERY.md, goal 1). This module is a
minimum faithful reading of Eq. (1) as written:

    PE[p, 2j]     = sin(p / n^(2j/d))
    PE[p, 2j + 1] = cos(p / n^(2j/d))
    Y(I)          = split(I) + PE(I)

with `p` the raster index of an image patch and `d` the channel dimension. The
manuscript applies PE to the image itself (not to token embeddings), so the
encoding is computed per patch and broadcast over that patch's pixels, keeping
the result in image space at [B, 3, H, W].

Two variants are provided; `sincos1d` is the literal reading of Eq. (1) and is
the default. `sincos2d` splits the channel budget between row and column index,
which preserves 2-D structure that a raster index discards. The choice is a
config field so the ablation can report it rather than hide it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _sincos_1d(num_patches: int, dim: int, n: float = 10000.0) -> torch.Tensor:
    """Literal Eq. (1): raster patch index -> `dim` channels. -> [num_patches, dim]"""
    p = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)      # [P, 1]
    j = torch.arange(dim, dtype=torch.float32).unsqueeze(0)              # [1, d]
    # 2j/d with j the *pair* index, so channels 0,1 share a frequency, 2,3 share, ...
    freq = torch.pow(n, (2.0 * torch.floor(j / 2.0)) / dim)
    angle = p / freq                                                     # [P, d]
    pe = torch.where(j % 2 == 0, torch.sin(angle), torch.cos(angle))
    return pe


def _sincos_2d(grid_h: int, grid_w: int, dim: int, n: float = 10000.0) -> torch.Tensor:
    """Row/column split variant. -> [grid_h * grid_w, dim]"""
    half = dim // 2
    rows = _sincos_1d(grid_h, max(half, 1), n)                # [H, half]
    cols = _sincos_1d(grid_w, max(dim - half, 1), n)          # [W, dim-half]
    pe = torch.cat(
        [
            rows.unsqueeze(1).expand(grid_h, grid_w, rows.shape[-1]),
            cols.unsqueeze(0).expand(grid_h, grid_w, cols.shape[-1]),
        ],
        dim=-1,
    )
    return pe.reshape(grid_h * grid_w, dim)


class PositionalEncodingRGB(nn.Module):
    """x -> clamp(x + gamma * PE), all in image space.

    Input  : [B, 3, H, W], float in [0, 1]
    Output : [B, 3, H, W], float in [0, 1]

    The PE map is a registered buffer, so it is deterministic, has zero trainable
    parameters, and is exported with the checkpoint.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        channels: int = 3,
        gamma: float = 0.1,
        pe_type: str = "sincos1d",
        n: float = 10000.0,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} not divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.gamma = float(gamma)
        self.pe_type = pe_type

        grid = img_size // patch_size
        if pe_type == "sincos1d":
            pe = _sincos_1d(grid * grid, channels, n)
        elif pe_type == "sincos2d":
            pe = _sincos_2d(grid, grid, channels, n)
        else:
            raise ValueError(f"unknown pe_type {pe_type!r}")

        # [P, C] -> [1, C, grid, grid] -> broadcast to pixels within each patch
        pe = pe.reshape(grid, grid, channels).permute(2, 0, 1).unsqueeze(0)
        pe = pe.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
        self.register_buffer("pe_map", pe, persistent=True)   # [1, C, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.pe_map.shape[-2:]:
            raise ValueError(
                f"PositionalEncodingRGB expects {tuple(self.pe_map.shape[-2:])}, got {tuple(x.shape[-2:])}"
            )
        return torch.clamp(x + self.gamma * self.pe_map, 0.0, 1.0)

    def extra_repr(self) -> str:
        return (
            f"img_size={self.img_size}, patch_size={self.patch_size}, "
            f"gamma={self.gamma}, pe_type={self.pe_type}"
        )
