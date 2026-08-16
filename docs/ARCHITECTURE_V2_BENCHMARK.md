# Architecture v2 — Inference Complexity Benchmark

> **FRAMING UPDATED 16 Aug 2026 — read `MAJOR_REVISION_TWO_METHOD_STRATEGY.md` first.**
> The revision now presents **two methods**, not a replacement. **C0 is Original
> AE-TFPE**, the reference proposed architecture; it is **retained in the run
> matrix** and is **not** rejected as a method. The "HARD REJECT" verdicts below
> were reached under a *replacement-selection* rule set that no longer applies —
> they now read as: C0's cost is an **RQ4 finding**, and C0/C1 are not viable
> **Efficient AE-TFPE** candidates. C2-28 is the **leading Efficient AE-TFPE
> candidate**, not a replacement architecture, and is not adopted.

**Date:** 16 August 2026 · **Status:** benchmark only. No training, no protocol re-freeze.
**Command:** `scripts/benchmark_architectures.py` · **Raw output:** `${OUTPUT_ROOT}/architecture_v2/benchmark.json`

Evaluates four candidate architectures against the frozen decision rules, to
inform — not to enact — a possible `revision-protocol-v2`. The frozen v1 arms are
untouched and were re-verified after every code change: `check_shapes.py` reports
**16/16 passing** with `A5_aetfpe_full` and `D1_ae_standard` still at exactly
**87,549,123** parameters.

---

## 1. Headline result

**C2 meets the preferred target on every criterion measurable without a T4. Per the
frozen decision rule, the recommendation is C2** — subject to one serious risk in
§7 that the decision rules do not cover and that must be resolved before re-freeze.

| | Params | × base | GFLOPs | × base | Verdict on cost |
|---|---|---|---|---|---|
| **BASELINE** YOLOv8n-cls | 1,488,247 | 1.00× | 0.4116 | 1.00× | reference |
| **C0 = Original AE-TFPE** | 87,549,123 | 58.83× | 36.2215 | **88.00×** | reference method; cost is an RQ4 finding |
| **C1** MobileViT-XXS + image AE | 2,700,147 | 1.81× | 3.0087 | **7.31×** | not viable as an Efficient candidate |
| **C2** MobileViT-XXS + slim feature AE | 2,545,603 | 1.71× | 1.1279 | **2.74×** | **meets preferred** |
| **C3** EfficientViT-B0 + slim feature AE | 2,264,323 | 1.52× | 0.8055 | **1.96×** | **meets preferred** |

Preferred: params ≤ ~3 M, GFLOPs ≤ 3×, latency ≤ 3×. Hard reject: GFLOPs > 5× or latency > 5×.

### The finding that matters most

**Replacing ViT-B/16 is necessary but not sufficient.** C1 swaps the 85.8 M
transformer for a 0.95 M one and is *still hard-rejected* at 7.31× — because the
image-space auto-encoder alone contributes ≈2.06 GFLOPs, a 5.0× floor on only
259 K parameters. Moving the AE to feature space drops it to ≈0.20 GFLOPs, a
**10.3× reduction**, and that single change is what carries C1 → C2 across the
threshold.

The auto-encoder's *operating space*, not the transformer's *size*, is the
binding constraint once the ViT is gone. Parameter count conceals this completely:
C1 and C2 differ by 6% in parameters and by **167%** in FLOPs.

---

## 2. Benchmark protocol

Identical for every candidate, including the baseline.

| Setting | Value |
|---|---|
| Input | 224 × 224 × 3 |
| Batch sizes | 1 and 32 |
| Warm-up iterations | 50 |
| Timed iterations | 200 |
| Synchronisation | `torch.cuda.synchronize()` around every timed region |
| AMP | identical across candidates (default: off, matching the training protocol) |
| Classifier | YOLOv8n-cls, unmodified, in all five rows |
| FLOPs convention | 2 × MACs, via `thop`, traced on CPU for device-independence |

**Parameters, FLOPs, tensor shapes and interfaces are hardware-independent and
final.** Latency, throughput and peak GPU memory are CUDA-only and are **not yet
measured** — see §8.

---

## 3. Per-candidate results

### BASELINE — YOLOv8n-cls only

| | |
|---|---|
| Total / trainable params | 1,488,247 / 1,488,247 |
| GFLOPs (MACs) | 0.4116 (0.2058 G) |
| Model size | 5.70 MB |
| Interface | `x [B,3,224,224] → classifier → [B,39]` |
| Transformer | none |
| AE | none |

### C0 — ViT-B/16 + image-space AE  *(the current frozen A5)*

| | |
|---|---|
| Total / trainable params | 87,549,123 / 1,750,467 |
| GFLOPs (MACs) | 36.2215 (18.1107 G) |
| Model size | 334.58 MB |
| Transformer | `google/vit-base-patch16-224-in21k`, **frozen**, 768-d tokens |
| AE space | **image** |
| Interface | `PE-RGB [B,3,224,224] ⊕ TF-RGB [B,3,224,224] → [B,6,224,224] → AE → latent [B,128,28,28] → recon [B,3,224,224] → YOLO` |
| Classifier stem | unmodified (3 ch) |

**Hard reject at 88.00× baseline FLOPs.** Also 1.60× the FLOPs of plain ViT-B/16
used directly as a classifier (22.571 GFLOPs) — the pipeline costs more than the
transformer inside it.

### C1 — MobileViT-XXS + image-space AE

| | |
|---|---|
| Total / trainable params | 2,700,147 / 1,749,123 |
| GFLOPs (MACs) | 3.0087 (1.5044 G) |
| Model size | 10.92 MB |
| Transformer | `mobilevit_xxs` (timm, ImageNet), **frozen**, 320-ch grid at 7×7 |
| AE space | **image** |
| Interface | `PE-RGB [B,3,224,224] ⊕ TF-RGB [B,3,224,224] → [B,6,224,224] → AE → latent [B,128,28,28] → recon [B,3,224,224] → YOLO` |
| Classifier stem | unmodified (3 ch) |

**Hard reject at 7.31× baseline FLOPs**, despite passing the parameter target at
1.81×. The image-space AE contributes ≈2.06 of the 3.01 GFLOPs — **68% of the
total cost sits in a 259 K-parameter module.**

### C2 — MobileViT-XXS + slim feature-space AE  ← recommended on cost

| | |
|---|---|
| Total / trainable params | 2,545,603 / 1,594,579 |
| GFLOPs (MACs) | 1.1279 (0.5640 G) |
| Model size | 10.33 MB |
| Transformer | `mobilevit_xxs` (timm, ImageNet), **frozen**, 320-ch grid at 7×7 |
| AE space | **feature** |
| Interface | `PE-RGB [B,3,224,224] --avgpool--> [B,3,7,7] ⊕ backbone [B,320,7,7] → [B,323,7,7] → AE encoder → latent [B,64,7,7] → decoder ×32 → recon [B,3,224,224] → YOLO` |
| Classifier stem | unmodified (3 ch) |

Meets both measurable preferred targets: 1.71× params, 2.74× FLOPs.

### C3 — EfficientViT-B0 + slim feature-space AE

| | |
|---|---|
| Total / trainable params | 2,264,323 / 1,581,715 |
| GFLOPs (MACs) | 0.8055 (0.4028 G) |
| Model size | 9.25 MB |
| Transformer | `efficientvit_b0` (timm, ImageNet), **frozen**, 128-ch grid at 7×7 |
| AE space | **feature** |
| Interface | `PE-RGB [B,3,224,224] --avgpool--> [B,3,7,7] ⊕ backbone [B,128,7,7] → [B,131,7,7] → AE encoder → latent [B,64,7,7] → decoder ×32 → recon [B,3,224,224] → YOLO` |
| Classifier stem | unmodified (3 ch) |

**C3 selection rationale.** `efficientvit_b0` (0.68 M backbone, 0.19 GFLOPs) is the
smallest ImageNet-pretrained global-context encoder available in timm. Everything
smaller in the surveyed families either has no pretrained weights or no attention
mechanism. Its multi-scale linear attention is a genuine global-context operator,
so it remains faithful to §3.2's "long-range contextual dependencies".

| Surveyed | Params | GFLOPs | Pretrained |
|---|---|---|---|
| `efficientvit_b0` | 0.68 M | 0.193 | ✓ `r224_in1k` |
| `mobilevit_xxs` | 0.95 M | 0.514 | ✓ `cvnets_in1k` |
| `mobilevitv2_050` | 1.11 M | 0.723 | ✓ |
| `edgenext_xx_small` | 1.16 M | 0.393 | ✓ |
| `xcit_nano_12_p16` | 2.53 M | 1.049 | ✓ |
| `vit_tiny_patch16_224` | 5.52 M | 2.149 | ✓ |

---

## 4. Frozen vs trainable, per candidate

Every candidate keeps the transformer **frozen**, matching frozen-protocol v1. The
trainable count is what the optimiser actually updates:

| Candidate | Total | Trainable | Frozen | Transformer |
|---|---|---|---|---|
| BASELINE | 1,488,247 | 1,488,247 | 0 | — |
| C0 | 87,549,123 | 1,750,467 | 85,798,656 | frozen |
| C1 | 2,700,147 | 1,749,123 | 951,024 | frozen |
| C2 | 2,545,603 | 1,594,579 | 951,024 | frozen |
| C3 | 2,264,323 | 1,581,715 | 682,608 | frozen |

Trainable counts sit near the 1.49 M baseline throughout, so training-step cost is
dominated by the frozen forward pass — which is exactly what Option A's feature
caching would remove.

---

## 5. Manuscript consistency of the feature-space AE

**Moving the AE to feature space makes §5.1 and §5.3 true. The image-space AE made
them false.** This is a correction *toward* the submitted manuscript.

> "Since the auto-encoder **operates on intermediate feature representations rather
> than raw images**, the number of parameters and floating-point operations (FLOPs)
> remains relatively low." — §5.1

C2/C3 encode at the backbone's 7×7 feature grid: literally an intermediate feature
representation, not raw images. And the FLOPs claim becomes true — 0.20 vs 2.06
GFLOPs.

> "the fusion mechanism operates on **fixed-dimensional latent features rather than
> directly on image pixels**." — §5.3

C2/C3 fuse at `[B,323,7,7]` (C2) or `[B,131,7,7]` (C3) — fixed-dimensional latent
features. The image-space variant fuses at `[B,6,224,224]`, i.e. directly on image
pixels, contradicting this sentence.

**However — §3.3 becomes less accurate, not more:**

> "First, **TF-RGB and PE-RGB** are input to the sparse autoencoder" — §3.3

In feature space, what is fused is the *raw backbone feature map* and an
*average-pooled PE-RGB*, not TF-RGB (the 3-channel image-space projection defined
in §3.2). §3.3 must be reworded to say the AE consumes the backbone's intermediate
representation together with the pooled positional map.

**Net:** the feature-space AE resolves a real internal contradiction (§3.3/Fig. 2
vs §5.1/§5.3) in favour of the efficiency claims, at the cost of one reworded
sentence in §3.3. That is a favourable trade, and it is disclosable as fixing an
inconsistency rather than changing a method.

---

## 6. Manuscript elements that must change

### Equations

| Element | Change | Why |
|---|---|---|
| **Eq. (2)** `Γ₀ = [i_class; i_p⁽¹⁾E; …] + E_pos` | Replace, or narrow to a citation | Describes ViT-style patch-embedding into `(N+1, D)` tokens with a CLS token. MobileViT and EfficientViT do not produce a flat token sequence with a class token; they emit a spatial grid `[B, C, 7, 7]`. Eq. (2) is false of C2/C3 as written. |
| **Eq. (3)** `Γ_l = Υ(Ψ(Γ_{l−1})) + Γ_{l−1}` | Keep, restate generically | Residual pre-norm blocks still describe both encoders. Must also fix the existing defect: both lines are currently printed identically. |
| **Eq. (4)** MHA `[Q,K,V] = ΓU` | Keep for C2, **revise for C3** | MobileViT uses standard spatial self-attention (Eq. 4 holds). EfficientViT uses multi-scale **linear** attention — Eq. 4's softmax formulation does not hold. Choosing C3 requires rewriting Eq. (4). |
| **Eq. (5)–(6)** AE objective | Keep; fix existing notation defects | The objective is unchanged by the operating space. The assignment-notation and `1=1` defects remain to be fixed regardless. |
| **New** | State the encoder configuration | The manuscript has *never* specified `D`, `L`, `N`, `P` or a variant. This is the moment to fill that blank. |

> **This is a real cost of C3 over C2.** C2 preserves Eq. (4); C3 invalidates it.
> Since C2 already meets the preferred target, C3's marginal FLOPs saving
> (1.96× vs 2.74×) does not obviously justify rewriting the attention equation.

### Figure 2

| Element | Change |
|---|---|
| "Linear Projection of Flattened Patches" + numbered patch-embedding column | Replace with the encoder's stage diagram — C2/C3 emit a spatial grid, not a flattened token sequence |
| "Transformer Encoder / L×" block | Keep; annotate with the actual depth and width once chosen |
| "Feature Fusion" box (Encoder → code → Decoder → image) | Redraw at the 7×7 grid; annotate the latent as `[64,7,7]` and mark the ⊕ as occurring at grid resolution |
| PE branch arrow into fusion | **Must show the average-pool to 7×7** — currently drawn as full-resolution |
| YOLOv8 input arrow | Keep; the classifier remains unmodified in every candidate |
| Any component sizes | Add — the figure currently states none |

### Prose

| Section | Change |
|---|---|
| §3.2 | State the encoder: variant, parameter count, grid, frozen status, pretrained source |
| §3.3 | Reword the AE's inputs (backbone feature map + pooled PE, not TF-RGB) and state the operating grid |
| §5.1 | Replace the qualitative paragraph with the measured table; the claim becomes true rather than asserted |
| §5.2 | Re-ground the deployment claim in the measured 2.5 M / 1.13 GFLOPs |
| §5.3 | Now accurate as written — no change needed once the AE moves |
| Abstract, §1 | Add the measured efficiency figure |

---

## 7. Blocking risk the decision rules do not cover

The rules score cost only. This one is about whether the architecture can work at all.

### The latent bottleneck is 32× harsher in C2/C3

| | Latent | Values | Input pixels | Compression |
|---|---|---|---|---|
| C0 / C1 (image-space) | `[128,28,28]` | 100,352 | 150,528 | **1.5×** |
| C2 / C3 (feature-space) | `[64,7,7]` | 3,136 | 150,528 | **48.0×** |

In C2/C3 the decoder must synthesise **150,528 pixel values from 3,136 latent
values**, upsampling 7×7 → 224×224. The reconstruction handed to YOLOv8n-cls
cannot carry fine lesion texture — precisely the signal a leaf-disease classifier
depends on. **Clean accuracy may collapse,** and gate G4 requires A5 clean
validation top-1 ≥ 0.95.

This is a direct consequence of putting the AE in feature space, and it is the
mechanism by which C2/C3 are cheap. The cost saving and the risk are the same fact.

**A second, related consequence:** PE-RGB is average-pooled from 224×224 to 7×7
before fusion — a 1024× reduction in spatial resolution. The positional encoding is
already a fixed additive field expected to produce a null result; pooling it to 7×7
makes its contribution inside A5 close to vacuous. The `A1_pe_only` ablation is
unaffected (no AE), but the PE branch's role in the full method would become
essentially decorative, and the manuscript should not claim otherwise.

### Options — **none applied**, per "do not optimize after observing results"

1. **Accept and test.** Run C2 through G4. If clean top-1 ≥ 0.95, the bottleneck is
   survivable and the result stands. Cheapest path to an answer; one short run.
2. **Widen the latent grid** (e.g. take a stride-16 stage → 14×14, latent
   `[64,14,14]` = 12,544, compression 12×). Costs FLOPs; still far below C1.
3. **Skip connection** from PE-RGB into the decoder. Changes fusion semantics — the
   instruction was not to change them, so this needs explicit approval.
4. **Feed the latent to YOLO directly**, dropping the decoder. Cheapest of all, but
   modifies the classifier stem, contradicting the §1 "without requiring
   modifications to the underlying classifiers" contribution.

Option 1 is recommended: it is a measurement, not a redesign, and it resolves the
question before anything is frozen.

---

## 8. What is still missing: T4 latency

Latency, throughput and peak GPU memory **have not been measured** and are not in
this document. They require a CUDA device; this machine has none, and the frozen
protocol forbids quoting non-CUDA timings as evidence. A CPU sanity run confirmed
the ordering (C0 ≫ C1 > C2 > C3) and that the harness works, but those numbers are
development output and are deliberately excluded here.

```bash
# COLAB — Tesla T4, completes the benchmark
python scripts/benchmark_architectures.py --device cuda --pretrained
```

This produces, for all five rows at batch 1 and 32: mean/median/std latency over
200 timed iterations after 50 warm-up, throughput, and peak GPU memory — and
evaluates the latency criteria automatically, writing
`verdict.latency_criterion_evaluated: true`.

**The cost verdicts in §1 will not change** — parameters and FLOPs are
hardware-independent. Only the latency columns are pending. C0 and C1 are already
hard-rejected on FLOPs alone, so the T4 run is decisive only between C2 and C3, and
for confirming C2 clears the 3× latency bar.

---

## 9. Recommendation

**Per the frozen decision rules: C2** — MobileViT-XXS + slim feature-space AE +
unmodified YOLOv8n-cls. It meets the preferred target on both measurable criteria
(1.71× params, 2.74× FLOPs, both inside the ≤3 M / ≤3× bounds), preserves Eq. (4),
keeps the classifier unmodified, and makes §5.1/§5.3 true.

C3 is cheaper still (1.52× / 1.96×) but requires rewriting Eq. (4) for linear
attention, and the rules select C3 only if C2 fails. C2 does not fail.

Option B (training-only auxiliary branches) is **not** triggered: it applies only
if both C2 and C3 exceed the hard-reject threshold, and neither does.

### Before any re-freeze, three things must happen

1. **Run the T4 benchmark** (§8) to close the latency criterion.
2. **Resolve the latent-bottleneck risk** (§7) — recommended via option 1, a single
   short C2 run checked against G4's clean-accuracy floor.
3. **Decide the §3.3 rewording and Fig. 2 redraw** (§6), since these change what the
   paper claims the AE consumes.

Only then is `revision-protocol-v2` justified. Nothing in v1 has been re-frozen,
and the 16 frozen arms remain byte-identical.

---

## Appendix — reproducing this benchmark

```bash
# hardware-independent figures (any device)
python scripts/benchmark_architectures.py --device cpu

# full benchmark including latency (CUDA required)
python scripts/benchmark_architectures.py --device cuda --pretrained

# confirm the frozen v1 arms are unaffected by the exploratory code
python scripts/check_shapes.py --device cpu   # expect 16/16, A5 = 87,549,123
```

Exploratory code added for this benchmark, all inert under v1 defaults:

| File | Purpose |
|---|---|
| `src/aetfpe/features/timm_encoder.py` | `TimmGlobalContextRGB` — lightweight encoders, same output contract as the ViT-B/16 branch |
| `src/aetfpe/autoencoder/model.py` | `SlimFeatureSpaceAE` appended; `StackedSparseDenoisingAE` unchanged |
| `src/aetfpe/models/aetfpe.py` | `tf_backbone`, `ae_space`, `ae_slim_*` config fields — defaults `""` / `"image"` reproduce v1 exactly |
| `scripts/benchmark_architectures.py` | This benchmark |

No config in `configs/` was modified; no frozen arm changed.
