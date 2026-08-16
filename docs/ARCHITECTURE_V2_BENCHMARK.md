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

**Parameters, tensor shapes and interfaces are hardware-independent and final.**
Latency, throughput and peak GPU memory were **measured on a Tesla T4 on
16 Aug 2026** — see §8.

⚠ **The FLOPs convention above is defective.** `thop` over-counts
`ConvTranspose2d` by stride², inflating every auto-encoder-bearing row. Corrected
figures are in §8; the derivation and empirical verification are in
`PROTOCOL_AMENDMENT_2026-08-16.md` §A2. The uncorrected values are retained
throughout §1–§7 as the historical record.

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

## 8. Official Tesla T4 latency — **MEASURED 16 Aug 2026**

Executed on Tesla T4 with `--batch-sizes 1 32 --warmup 50 --iters 200`, all five
models in one process on one GPU. These supersede the "not yet measured" status
this section previously carried. **Read `PROTOCOL_AMENDMENT_2026-08-16.md` before
quoting any verdict from this run.**

Run provenance from the JSON: commit `3d5f1c4`, clean tree, Tesla T4, torch
2.11.0+cu128 / CUDA 12.8, `device_audit.verified: true` for every candidate.
**Note the JSON names the 7×7 candidate `C2`, not `C2-7`.**

| Candidate | BS1 latency | × base | BS1 img/s | BS32 per-image | × base | BS32 img/s | Peak MiB BS1 | Peak MiB BS32 |
|---|---|---|---|---|---|---|---|---|
| **BASELINE** YOLOv8n-cls | **3.3632 ms** | 1.000 | 297.34 | **0.22906 ms** | 1.000 | 4365.58 | 41.61 | 130.61 |
| **C0 = Original AE-TFPE** | 28.5088 ms | **8.477×** | 35.08 | 10.7734 ms | **47.033×** | 92.82 | 378.12 | 646.25 |
| **`C2` = C2-7** | 27.0000 ms | **8.028×** | 37.04 | 1.2318 ms | 5.378× | 811.74 | 50.53 | 273.07 |
| **C2-14** | 16.3708 ms | **4.868×** | 61.08 | 1.2260 ms | 5.352× | 815.63 | 58.54 | 281.09 |
| **C2-28** | 10.2045 ms | **3.034×** | 98.00 | 1.2908 ms | 5.635× | 774.78 | 65.21 | 287.75 |

⚠ **Batch-1 latency is noisy: 15–30 % relative standard deviation.** C2-28's mean
ratio is 3.034× but its median ratio is **2.923×** — the two straddle the 3×
threshold, and the standard deviation (2.04 ms) is ~60× the gap to it. The rule
does not specify mean or median. See amendment §A5: **no claim may rest on the 3×
boundary until batch-1 latency is re-measured with lower variance.** Batch-32
timings are tight (0.9–2.7 %) and need no such caveat.

### The result that inverts the FLOPs ordering

**C2-7 has the fewest FLOPs and nearly the worst latency.** It is 2.6× slower than
C2-28 at batch 1 while computing *less* arithmetic. FLOPs are a poor proxy here:
the 7×7 stage runs a 320-channel 1×1 encoder on a 49-pixel grid and carries two
extra bilinear interpolations inside MobileViT's stage-4 block (`DECODER_PATH_AUDIT.md`
§C). Small tensors, many kernel launches — latency-bound, not arithmetic-bound.

**This vindicates measuring latency on the real device rather than inferring it.**
Any selection made on FLOPs alone would have chosen C2-7, the slowest viable
candidate at batch 1.

### Two audits were commissioned on this run — both found defects

| Audit | Finding | Where |
|---|---|---|
| Decision-rule consistency | The latency criterion is bound to `args.batch_sizes[0]` — the verdict depends on **CLI argument order**, and the pre-registered rule never named a batch size | `PROTOCOL_AMENDMENT_2026-08-16.md` §A1 |
| Decoder shape path | `describe()` misreports decoder depth; the architecture itself is correct and contains **no interpolation** in any decoder | `DECODER_PATH_AUDIT.md`, amendment §A3 |
| *(found during the above)* | `thop` over-counts `ConvTranspose2d` by stride² = 4×, inflating every AE-bearing candidate's GFLOPs | amendment §A2 |
| *(found during the above)* | `tf.project` / `tf.norm` are dead parameters in every C2-* candidate | amendment §A4 |

**The `verdict` field in `t4_benchmark.json` must not be quoted** until §A1 is
resolved. The latency measurements above are unaffected and stand.

### Corrected FLOPs — **[DERIVED]**, see amendment §A2

`thop`'s transposed-convolution over-count was verified empirically against
ground truth. Corrected, uniformly, for every candidate:

| Candidate | GFLOPs as reported | × base | GFLOPs corrected | × base |
|---|---|---|---|---|
| BASELINE | 0.4116 | 1.000 | 0.4116 | 1.000 |
| C0 = Original AE-TFPE | 36.2215 | 88.00× | **34.8728** | **84.73×** |
| C2-7 | 1.1279 | 2.740× | **0.9786** | **2.378×** |
| C2-14 | 1.3896 | 3.376× | **1.0043** | **2.440×** |
| C2-28 | 1.8215 | 4.425× | **1.0123** | **2.459×** |

The three grids are **near-identical in FLOPs** (2.38 / 2.44 / 2.46) once the
over-count is removed. FLOPs do not separate them; measured BS1 latency does.
Any argument in `ARCHITECTURE_V2_SPATIAL_TRADEOFF.md` that rests on the
2.74 → 3.38 → 4.43 progression must be rewritten.

---

## 9. Standing of C2-28 after the T4 run

Superseded the pre-T4 recommendation below. Under the **proposed** BS1-primary
clarification (amendment §A1 — not yet adopted), C2-28 is classified criterion by
criterion:

| Criterion | Measured | Classification |
|---|---|---|
| **Parameters** | 1,716,739 · 1.15× baseline · ≤ 3 M | **PREFERRED** |
| **FLOPs** | 4.425× as reported, **2.459× corrected** | **CONDITIONAL** — depends on adopting §A2 |
| **BS1 latency** *(primary)* | 10.2045 ms · **3.034× mean, 2.923× median** vs a 3.0× threshold | **INDETERMINATE** — mean and median straddle the threshold (§A5) |
| **BS32 throughput** *(secondary)* | 5.635× per-image · 774.78 vs 4365.58 img/s | **SUBSTANTIAL OVERHEAD** relative to baseline |
| **Model size** | 7.156 MB · 1.255× baseline | preferred |
| **Peak CUDA memory** | 65.21 MiB BS1 (1.57× base) · 287.75 MiB BS32 (2.20× base) | moderate overhead; **5.80× / 2.25× lower than Original AE-TFPE** |

### Overall: **CONDITIONAL CANDIDATE — NOT YET VALIDATED**

C2-28 remains the **leading Efficient AE-TFPE candidate**. It is **not** adopted,
**not** validated, and **not** shown to be accurate — no C2-28 model has been
trained. `meets_preferred` is **false** under every accounting in this document.

**C2-28 must not be described as "efficient" on the strength of this benchmark.**
It has been measured for cost only. The phrases **"negligible overhead"** and
**"baseline-equivalent efficiency"** are forbidden: they are contradicted by
3.034× BS1 latency and 5.635× BS32 per-image cost.

The defensible statement is:

> **C2-28 substantially reduces the computational burden of Original AE-TFPE, but
> still incurs meaningful overhead over YOLOv8n-cls.**

### Before any re-freeze

1. ~~Run the T4 benchmark~~ — **done**, §8.
2. **Resolve the five decisions** in `PROTOCOL_AMENDMENT_2026-08-16.md`, starting
   with the unreconciled C2-7 verdict value.
3. **Resolve the latent-bottleneck risk** (§7) via the single C2-28 clean-validation
   sanity run — **prepared, not executed**.
4. **Resolve the determinism finding** (`MAJOR_REVISION_TWO_METHOD_STRATEGY.md` §9a).
5. **Decide the §3.3 rewording and Fig. 2 redraw** (§6).

Only then is `revision-protocol-v2` justified. Nothing in v1 has been re-frozen,
and the 16 frozen arms remain byte-identical.

### Superseded pre-T4 recommendation *(kept for the audit trail)*

> Per the frozen decision rules: C2 — it meets the preferred target on both
> measurable criteria (1.71× params, 2.74× FLOPs). C3 is cheaper still (1.52× /
> 1.96×) but requires rewriting Eq. (4) for linear attention. Option B is not
> triggered.

That recommendation was reached **before** any latency existed, and on FLOPs that
are now known to be inflated. The T4 run shows FLOPs would have selected C2-7 —
the slowest viable candidate at batch 1. Retained as evidence of what the
pre-registered rules produced at the time, not as a live recommendation.

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
