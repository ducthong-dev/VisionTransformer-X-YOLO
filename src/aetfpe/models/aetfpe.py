"""The canonical AE-TFPE model: front-end + unmodified classifier.

Every ablation arm is this one class with different flags, so the arms cannot
drift apart. The front-end always emits an image-space tensor, which is what
lets the classifier stay stock:

    x  [B,3,224,224] in [0,1]
      -> PE-RGB   [B,3,224,224]        (use_pe, else identity)
      -> TF-RGB   [B,3,224,224]        (use_tf, else None)
      -> fusion   [B,3 or 6,224,224]
      -> AE       [B,3,224,224]        (use_ae, else fusion output)
      -> classifier                    -> [B, num_classes]

The only arm that touches the classifier is plain concatenation, which needs a
6-channel stem. `AETFPE.classifier_in_channels` records what was actually built.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..autoencoder.model import StackedSparseDenoisingAE
from ..features.legacy_lut import legacy_transform_tensor
from ..features.positional_encoding import PositionalEncodingRGB
from ..features.transformer_features import TransformerFeatureRGB
from ..fusion.ops import build_fusion
from .classifier import build_classifier, classifier_forward


@dataclass
class AETFPEConfig:
    # --- front-end switches ---
    use_pe: bool = False
    use_tf: bool = False
    use_ae: bool = False
    fusion: str = "identity"          # add | concat | linear | attention | identity
    legacy_lut: bool = False          # arm A2: historical transform, no PE/TF/AE
    photometric: str = ""             # arm A3: e.g. "gamma:1.6"

    # --- component hyperparameters ---
    img_size: int = 224
    patch_size: int = 16
    pe_gamma: float = 0.1
    pe_type: str = "sincos1d"
    vit_name: str = "google/vit-base-patch16-224-in21k"
    vit_pretrained: bool = True
    vit_frozen: bool = True
    ae_latent_channels: int = 128
    ae_widths: tuple = (32, 64)
    ae_sparse: bool = True            # False -> plain (non-sparse) AE, arm D1
    ae_denoising: bool = True         # False -> clean-in/clean-out, arm D1

    # --- classifier ---
    classifier: str = "yolov8n-cls"
    num_classes: int = 39
    pretrained: bool = True

    meta: dict = field(default_factory=dict)


def _apply_photometric(x: torch.Tensor, spec: str) -> torch.Tensor:
    """Monotonic pointwise control transform (arm A3).

    Deliberately monotonic: it is matched to the legacy LUT in deviation
    magnitude but has none of its wrap-around discontinuities, which is exactly
    the comparison that isolates what the legacy transform's effect comes from.
    """
    if not spec:
        return x
    kind, _, value = spec.partition(":")
    v = float(value) if value else 1.0
    if kind == "gamma":
        return torch.clamp(x, 1e-6, 1.0).pow(v)
    if kind == "contrast":
        return torch.clamp((x - 0.5) * v + 0.5, 0.0, 1.0)
    raise ValueError(f"unknown photometric spec {spec!r}")


class AETFPE(nn.Module):
    def __init__(self, cfg: AETFPEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.pe = (
            PositionalEncodingRGB(
                img_size=cfg.img_size,
                patch_size=cfg.patch_size,
                gamma=cfg.pe_gamma,
                pe_type=cfg.pe_type,
            )
            if cfg.use_pe
            else None
        )
        self.tf = (
            TransformerFeatureRGB(
                model_name=cfg.vit_name,
                out_channels=3,
                freeze=cfg.vit_frozen,
                pretrained=cfg.vit_pretrained,
            )
            if cfg.use_tf
            else None
        )

        if cfg.use_ae:
            # The AE always consumes the *concatenated* map when both branches
            # exist, so AE fusion is a genuine alternative to the operators below
            # rather than a wrapper around one of them. `self.fusion` is not
            # built at all in this case: an earlier version built it
            # unconditionally and called it in forward() regardless of use_ae,
            # which computed a full fusion forward pass every step and then
            # discarded the result (`out = recon` always overrode it). That
            # left `fusion.*` as trainable parameters with permanently-None
            # gradients -- caught by G1's backward-pass check. Not building the
            # module at all removes the dead compute and the dead parameters;
            # it does not change any arm's actual forward computation, since
            # `fused` was never used when use_ae=True.
            self.fusion = None
            self.ae_in_channels = 6 if cfg.use_tf else 3
            self.ae = StackedSparseDenoisingAE(
                in_channels=self.ae_in_channels,
                out_channels=3,
                widths=tuple(cfg.ae_widths),
                latent_channels=cfg.ae_latent_channels,
            )
            classifier_in = 3
        else:
            fusion_name = cfg.fusion if cfg.use_tf else "identity"
            self.fusion = build_fusion(fusion_name, in_channels=6)
            self.ae = None
            self.ae_in_channels = 0
            classifier_in = self.fusion.out_channels

        self.classifier_in_channels = classifier_in
        self.classifier = build_classifier(
            cfg.classifier,
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
            in_channels=classifier_in,
        )

    # ------------------------------------------------------------------ #

    def frontend(self, x: torch.Tensor, return_parts: bool = False):
        """x in [0,1] -> tensor handed to the classifier.

        Returns (out, parts) where parts carries the pre-AE fused map and the AE
        latent, both needed by the latent-stability analysis (Phase 7).
        """
        parts: dict[str, torch.Tensor | None] = {"pre_ae": None, "latent": None}

        if self.cfg.legacy_lut:
            out = legacy_transform_tensor(x)
            parts["pre_ae"] = out
            return (out, parts) if return_parts else out

        if self.cfg.photometric:
            x = _apply_photometric(x, self.cfg.photometric)

        pe = self.pe(x) if self.pe is not None else x

        if self.tf is not None:
            tf = self.tf(pe)
            fused_for_ae = torch.cat([pe, tf], dim=1)
        else:
            tf = None
            fused_for_ae = pe

        parts["pre_ae"] = fused_for_ae

        if self.ae is not None:
            recon, latent = self.ae(fused_for_ae, return_latent=True)
            parts["latent"] = latent
            out = recon
        else:
            # self.fusion is None only when use_ae is True, and that branch is
            # handled above, so self.fusion is always built here.
            out = self.fusion(pe, tf) if tf is not None else pe

        return (out, parts) if return_parts else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return classifier_forward(self.classifier, self.frontend(x))

    # ------------------------------------------------------------------ #

    def frontend_parameters(self):
        for m in (self.pe, self.tf, self.fusion, self.ae):
            if m is not None:
                yield from m.parameters()

    def describe(self) -> dict:
        d = {
            "use_pe": self.cfg.use_pe,
            "use_tf": self.cfg.use_tf,
            "use_ae": self.cfg.use_ae,
            "fusion": (
                "ae" if self.cfg.use_ae and self.cfg.use_tf
                else self.cfg.fusion if self.cfg.use_tf
                else "identity"
            ),
            "legacy_lut": self.cfg.legacy_lut,
            "photometric": self.cfg.photometric or None,
            "classifier": self.cfg.classifier,
            "classifier_in_channels": self.classifier_in_channels,
            "classifier_stem_modified": self.classifier_in_channels != 3,
            "pretrained_transfer": getattr(self.classifier, "_pretrained_transfer", "?"),
            "num_classes": self.cfg.num_classes,
        }
        if self.tf is not None:
            d["transformer"] = self.tf.describe()
        if self.ae is not None:
            d["autoencoder"] = self.ae.describe()
            d["autoencoder"]["sparse"] = self.cfg.ae_sparse
            d["autoencoder"]["denoising"] = self.cfg.ae_denoising
        return d


def build_model(cfg_dict: dict) -> AETFPE:
    known = {f for f in AETFPEConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in cfg_dict.items() if k in known}
    return AETFPE(AETFPEConfig(**kwargs))
