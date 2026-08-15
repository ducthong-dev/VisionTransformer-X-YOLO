"""Fusion operators between PE-RGB and TF-RGB.

Every operator has the same signature -- two [B, 3, H, W] maps in, one
[B, C_out, H, W] map out -- so swapping them is a config change and the rest of
the pipeline is untouched. `out_channels` is 3 for every operator except
`concat`, which is the only one that forces a change to the classifier stem.
That exception is deliberate and is reported in the results table rather than
hidden (Reviewer #12 asks for plain concatenation specifically).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AddFusion(nn.Module):
    """F = 0.5 * (F_PE + F_TF).  [B,3,H,W] -> [B,3,H,W].  0 parameters."""

    out_channels = 3

    def forward(self, pe: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        return 0.5 * (pe + tf)


class ConcatFusion(nn.Module):
    """F = [F_PE ; F_TF].  [B,3,H,W] x2 -> [B,6,H,W].  0 parameters.

    Requires a 6-channel classifier stem; see models.stem.adapt_stem.
    """

    out_channels = 6

    def forward(self, pe: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        return torch.cat([pe, tf], dim=1)


class LinearProjectionFusion(nn.Module):
    """F = sigmoid(BN(W [F_PE ; F_TF] + b)), W a 1x1 conv 6 -> 3.

    The sigmoid is not decorative: every arm must hand the classifier a tensor in
    [0, 1], the same range the plain-RGB baseline supplies and the range the
    pretrained stem was trained for. An unbounded activation here would give this
    arm different input statistics from the baseline, which would confound the
    ablation with an artefact of normalisation.
    """

    out_channels = 3

    def __init__(self, in_channels: int = 6, out_channels: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.Sigmoid()
        self.out_channels = out_channels

    def forward(self, pe: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(torch.cat([pe, tf], dim=1))))


class AttentionFusion(nn.Module):
    """Squeeze-and-excitation channel gate over [F_PE ; F_TF], then 1x1 to 3 channels.

    s = sigmoid(W2 ReLU(W1 GAP([F_PE ; F_TF])))
    F = SiLU(BN(W ( s * [F_PE ; F_TF] )))

    A lightweight attention baseline is used deliberately in place of full
    cross-attention: at 196 tokens and image resolution, cross-attention costs
    more than the auto-encoder it is meant to be a cheap foil for, which would
    make the complexity comparison incoherent. The substitution is stated in the
    results table.
    """

    out_channels = 3

    def __init__(self, in_channels: int = 6, out_channels: int = 3, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.Sigmoid()          # range-matched to the baseline; see LinearProjectionFusion
        self.out_channels = out_channels

    def forward(self, pe: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pe, tf], dim=1)
        return self.act(self.norm(self.proj(x * self.gate(x))))


class IdentityFusion(nn.Module):
    """Pass-through for single-branch variants (RGB only, PE only, TF only)."""

    out_channels = 3

    def forward(self, pe: torch.Tensor, tf: torch.Tensor | None = None) -> torch.Tensor:
        return pe if tf is None else pe


FUSION_REGISTRY = {
    "add": AddFusion,
    "concat": ConcatFusion,
    "linear": LinearProjectionFusion,
    "attention": AttentionFusion,
    "identity": IdentityFusion,
    # "ae" is not listed here: the auto-encoder consumes the *concatenated* map,
    # so it is built by models.aetfpe rather than by this registry.
}


def build_fusion(name: str, in_channels: int = 6, **kwargs) -> nn.Module:
    if name not in FUSION_REGISTRY:
        raise KeyError(f"unknown fusion {name!r}; available: {sorted(FUSION_REGISTRY)}")
    cls = FUSION_REGISTRY[name]
    if cls in (AddFusion, ConcatFusion, IdentityFusion):
        return cls()
    return cls(in_channels=in_channels, **kwargs)
