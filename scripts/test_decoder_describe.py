#!/usr/bin/env python
"""Regression test: AE metadata must match the decoder that actually runs.

Guards protocol amendment A3. `SlimFeatureSpaceAE.describe()` previously derived
decoder depth from `len(self.decoder) // 2`, which counts Sequential entries
rather than upsampling stages and reported 3/2/2 for C2-7/C2-14/C2-28 whose real
depths are 5/4/3. The architecture was correct; only the description was wrong.

The test is deliberately not a tautology: every claim in `describe()` is checked
against shapes captured from a real forward pass through the decoder, so metadata
that drifts from the executed graph fails here.

    LOCAL / COLAB   python scripts/test_decoder_describe.py

CPU, scratch weights, no dataset, no downloads, no training. Exit 0 = all pass.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from aetfpe.models import build_model  # noqa: E402

# The verified paths, stated independently of the implementation.
EXPECTED = {
    "C2-7":  {"stage": None, "stages": 5, "path": [7, 14, 28, 56, 112, 224]},
    "C2-14": {"stage": 3,    "stages": 4, "path": [14, 28, 56, 112, 224]},
    "C2-28": {"stage": 2,    "stages": 3, "path": [28, 56, 112, 224]},
}

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def build(stage) -> torch.nn.Module:
    cfg = dict(use_pe=True, use_tf=True, use_ae=True, fusion="linear",
               tf_backbone="mobilevit_xxs", ae_space="feature",
               num_classes=39, img_size=224, pretrained=False, vit_pretrained=False)
    if stage is not None:
        cfg["tf_stage"] = stage
    return build_model(cfg).eval()


def observed_path(model) -> list[int]:
    """Spatial sizes actually produced by the decoder's transposed convolutions."""
    seen: list[int] = []

    def hook(m, inputs, out):
        if not seen:
            seen.append(int(inputs[0].shape[-1]))
        seen.append(int(out.shape[-1]))

    handles = [m.register_forward_hook(hook)
               for m in model.ae.decoder.modules() if isinstance(m, nn.ConvTranspose2d)]
    with torch.no_grad():
        model(torch.rand(1, 3, 224, 224))
    for h in handles:
        h.remove()
    return seen


def main() -> int:
    for label, exp in EXPECTED.items():
        print(f"\n[{label}]")
        model = build(exp["stage"])
        d = model.ae.describe()

        # 1. Metadata matches the independently stated ground truth.
        check(f"{label}: decoder_stages == {exp['stages']}",
              d["decoder_stages"] == exp["stages"], f"got {d['decoder_stages']}")
        check(f"{label}: decoder_spatial_path == {exp['path']}",
              d["decoder_spatial_path"] == exp["path"], f"got {d['decoder_spatial_path']}")
        check(f"{label}: decoder_output_size == 224", d["decoder_output_size"] == 224,
              f"got {d['decoder_output_size']}")

        # 2. Metadata matches what the decoder actually executes. This is the check
        #    that catches drift; the derivation could otherwise be self-consistent
        #    and still wrong.
        obs = observed_path(model)
        check(f"{label}: described path == observed forward path", obs == exp["path"],
              f"observed {obs}")

        # 3. The stage count equals the number of transposed convolutions built.
        n_ct = sum(1 for m in model.ae.decoder.modules() if isinstance(m, nn.ConvTranspose2d))
        check(f"{label}: stage count == ConvTranspose2d modules built",
              d["decoder_stages"] == n_ct, f"metadata {d['decoder_stages']} vs modules {n_ct}")

        # 4. The old defective expression must not be what is reported.
        stale = len(model.ae.decoder) // 2
        check(f"{label}: no longer reports len(decoder)//2 = {stale}",
              d["decoder_stages"] != stale or exp["stages"] == stale,
              f"stale value {stale}, correct {exp['stages']}")

        # 5. The human-readable string carries the real depth and full path.
        text = d["decoder"]
        check(f"{label}: summary string states {exp['stages']} stages",
              text.startswith(f"{exp['stages']} x "), text)
        check(f"{label}: summary string carries the full path",
              " -> ".join(str(v) for v in exp["path"]) in text, text)

        # 6. Metadata must not have disturbed the model.
        check(f"{label}: classifier still receives [1,3,224,224]",
              tuple(model.frontend(torch.rand(1, 3, 224, 224)).shape) == (1, 3, 224, 224))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
