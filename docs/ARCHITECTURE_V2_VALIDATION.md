# Architecture v2 — C2 Validation

**Date:** 16 August 2026 · **Candidate:** C2 = MobileViT-XXS + slim feature-space AE + unmodified YOLOv8n-cls
**Status:** C2-1 **not executed** (no CUDA on this machine) · C2-2 **executed** · C2-3 **blocked by the C2-1 gate**
**Verdict:** **CONDITIONAL FAIL — do not adopt C2 as specified.** See §7.

Nothing was trained. No protocol was re-frozen. The 16 frozen v1 arms are unchanged
(`check_shapes.py` 16/16, A5/D1 still 87,549,123 params).

---

## 1. Stage C2-1 — real CUDA efficiency · NOT EXECUTED

This machine has no CUDA device (`torch.cuda.is_available() == False`). The brief
says *"Do not estimate missing values"*, so latency, throughput and peak CUDA memory
are **absent from this document rather than estimated**.

Because you gated C2-3 behind C2-1 and C2-2, and C2-1 cannot run here, **C2-3 is
blocked** — see §5.

The harness is committed and implements the exact protocol (batch 1 and 32, 50
warm-up, 200 timed iterations, `torch.cuda.synchronize()`, identical AMP):

```bash
# COLAB / Tesla T4
python scripts/benchmark_architectures.py --device cuda --pretrained
```

It writes `${OUTPUT_ROOT}/architecture_v2/benchmark.json` with per-candidate
`latency_ms_mean/std/median`, `throughput_img_per_s`, `peak_gpu_mem_mb`, and
evaluates the latency rules automatically (`verdict.latency_criterion_evaluated:
true`).

**Already final and hardware-independent** (from the committed benchmark):

| | Params | × base | GFLOPs | × base |
|---|---|---|---|---|
| BASELINE | 1,488,247 | 1.00× | 0.4116 | 1.00× |
| C2 | 2,545,603 | 1.71× | 1.1279 | 2.74× |
| C3 | 2,264,323 | 1.52× | 0.8055 | 1.96× |

Only the latency columns are outstanding. Note that a 2.74× FLOPs ratio does not
guarantee a ≤3× latency ratio: MobileViT is memory-bandwidth-bound and historically
under-performs its FLOP count on a T4, so C2 could plausibly land above 3× latency
while passing on FLOPs. **The latency rule is genuinely undecided.**

---

## 2. Stage C2-2 — information-preservation audit · EXECUTED

Deterministic subset: **468 validation images**, 12 per class × 39 classes, selected
by sorted class name and sorted filename (no randomness). Script:
`scripts/audit_bottleneck.py`. Raw output:
`${OUTPUT_ROOT}/architecture_v2/bottleneck_audit/bottleneck_audit.json`.

### 2.1 Why no reconstruction MSE/PSNR/SSIM is reported

The brief asks for reconstruction metrics on the AE output. **At this stage the
auto-encoder is randomly initialised.** Measured directly: its decoder emits a
near-constant field with per-image mean **0.497** and across-image variation
**0.017**. MSE/PSNR/SSIM computed on that would characterise the random
initialisation, not the architecture — and would look like evidence while being
none. They are therefore omitted deliberately, not overlooked.

Two substitutes that *are* valid without training are reported instead.

### 2.2 Spatial-resolution floor

What a purely **spatial** grid of each size can carry, measured by
downsample→upsample against the original (468 images):

| Grid | Cell covers | MSE | PSNR | SSIM |
|---|---|---|---|---|
| **7×7** (C2's latent grid) | 32×32 px | 0.01153 | **20.04 dB** | **0.383** |
| 28×28 (C0's latent grid) | 8×8 px | 0.00467 | 24.09 dB | 0.462 |

**Direction of this bound:** C2's latent carries 64 channels (3,136 dims), far more
than a 3-channel thumbnail's 147, so this is a **pessimistic floor**, not a ceiling.
It shows what 7×7 *spatial support* costs, and nothing more.

But the spatial argument is the binding one, and it is geometric rather than
statistical: **at 7×7, one cell covers 32×32 pixels.** Septoria and rust lesions are
roughly 5–15 px — about 3× smaller than a single cell. No number of channels
restores *where inside a cell* a spot sat; a conv decoder without skip connections
can only render the cell's aggregate content.

### 2.3 Qualitative grid

`spatial_floor_grid.png` — original / 28×28 floor / 7×7 floor / |error| at 7×7, for
ten representative classes across the five requested categories.

| Category | Survives 7×7? |
|---|---|
| Healthy leaves (Apple, Tomato) | **Yes** — silhouette and colour intact |
| Large lesions (Late blight, Black rot) | **Partly** — coarse blob, boundaries gone |
| Small localized spots (Cedar apple rust, Septoria) | **No** — spots vanish entirely |
| Texture-dominated (Squash / Cherry powdery mildew) | **No** — texture erased |
| Visually similar pair (Early blight vs Bacterial spot) | **No** — the distinguishing detail is what is lost |

The error maps concentrate on exactly the disease-relevant structures: lesion
boundaries, leaf margins, and veins. **The 7×7 floor preserves what the classes have
in common and discards what separates them.**

### 2.4 Class separability at the bottleneck

The decoder is a deterministic function of the latent, so it can only distinguish
images whose latents differ. Measuring latent separability therefore bounds what any
decoder could preserve. Cosine 1-NN (5-fold CV) and silhouette, chance = 0.0256:

| Representation | Dims | 1-NN accuracy | Silhouette | cos(imgᵢ, imgⱼ) |
|---|---|---|---|---|
| F_RGB (28×28 pooled pixels) | 2,352 | 0.2223 ± 0.027 | −0.191 | 0.896 |
| F_PE (3×7×7) | 147 | 0.2459 ± 0.034 | −0.220 | 0.927 |
| F_TF (320×7×7) | 15,680 | 0.1475 ± 0.029 | −0.170 | 0.500 |
| **Z_AE (64×7×7)** | 3,136 | **0.1839 ± 0.027** | **−0.060** | 0.755 |

**Channel compression is not the problem.** Z_AE (64 ch) separates classes at least
as well as F_TF (320 ch) — 0.184 vs 0.148, with the best silhouette of the four. The
AE encoder here is a random 1×1 projection 323→64; by Johnson–Lindenstrauss such a
projection approximately preserves pairwise distances, so this is a **lower bound**
on what a trained encoder achieves. The 323→64 bottleneck is safe.

**Caveats, stated because these numbers are weak in absolute terms.** All silhouettes
are negative: frozen representations overlap heavily, which is expected — a trained
classifier on top reaches ~99%. Cosine 1-NN on *unnormalised* conv activations is
also a poor probe (F_TF's raw dynamic range lets a few channels dominate), which is
the likely reason frozen MobileViT features score below raw pooled pixels here. These
numbers support the **relative** conclusion (64 ch ≈ 320 ch) and should not be read
as absolute capability.

---

## 3. PE branch information audit · EXECUTED

Two findings. The first concerns PE itself and applies to every candidate. The second
is specific to C2 and is a defect.

### 3.1 PE's contribution is 99.7% constant — a scientific finding

`PE-RGB = clamp(x + γ·PE_map)` with γ = 0.1, and `PE_map` is a fixed buffer. Measuring
the PE-induced change at the fusion input, decomposed into its constant component and
its image-dependent remainder:

| Quantity | Value |
|---|---|
| PE's total effect at the fusion input, ‖Δ‖ | 0.48991 |
| Constant component of that effect | 0.48848 |
| **Constant fraction** | **99.71%** |
| Image-dependent fraction | **0.29%** |

PE also perturbs the transformer's features by 21.6% ± 16.8% relative — not
negligible in magnitude — but that perturbation is **the same field for every image**,
so it acts as a fixed input bias rather than as information. Relative to between-image
variation in F_TF (0.988), PE's effect is 0.218 — a real distortion carrying
essentially no image-dependent signal.

**Recorded as a finding, not rescued.** This converges with the pre-registered
expectation that `A1_pe_only ≈ A0_baseline_rgb`. If the ablation confirms it, the
manuscript cannot continue to claim PE-RGB as one of three *complementary*
contributing components (Abstract, §1, §3.1). No change to PE has been made.

### 3.2 In C2 the PE branch is additionally swamped 243:1 — a C2-specific defect

| | Dims | ‖·‖ | Per-element RMS |
|---|---|---|---|
| F_PE (pooled PE-RGB, values in [0,1]) | 147 | 5.95 | **0.491** |
| F_TF (raw MobileViT activations, unnormalised) | 15,680 | 14,943.53 | **119.34** |

**F_TF is 243× larger per element than F_PE**, and F_PE is 0.93% of the fused
channels and **0.040% of the fused magnitude**.

This is not a property of PE — it is a property of how C2 fuses. The image-space
path (C0/C1) concatenates two sigmoid-bounded `[0,1]` maps at 224², so PE is 50% of
channels at a comparable scale. Moving to feature space concatenates a `[0,1]`
thumbnail with unnormalised backbone activations, and the AE's 1×1 conv sees the PE
channels as numerical noise.

**So C2 does not merely weaken the PE branch; it deletes it.** Adopting C2 means the
paper can no longer claim PE-RGB as a component of the deployed method, independently
of whether PE would have worked at full resolution.

Fixing this would require normalising the backbone features before concatenation —
a change to fusion semantics, which the brief forbids and which I have not made.

---

## 4. Combined C2-2 assessment

| Sub-question | Verdict |
|---|---|
| Does the 323→64 **channel** compression destroy class information? | **No** — Z_AE ≥ F_TF separability |
| Does the 7×7 **spatial** grid preserve small lesions, boundaries, texture? | **No** — geometrically impossible below a 32×32 cell |
| Is the fused representation still class-separable at the bottleneck? | **Partly** — above chance, but weak and unimproved over raw pixels |
| Does the PE branch carry image-dependent information? | **No** — 99.7% constant |
| Is the PE branch numerically present in C2's fusion at all? | **No** — 0.040% of magnitude, swamped 243:1 |
| Can YOLO classify the decoder's rendering? | **UNTESTABLE without training** — this is C2-3 |

---

## 5. Stage C2-3 — classification sanity gate (G4) · NOT EXECUTED

Blocked twice over: your sequencing requires C2-1 and C2-2 to pass first, C2-1 cannot
run on this machine, and C2-2 did not cleanly pass (§4).

When it does run, the command is:

```bash
python scripts/train.py --config configs/aetfpe_full.yaml \
    --override model.tf_backbone=mobilevit_xxs \
    --override model.ae_space=feature \
    --out "$OUTPUT_ROOT/validation/C2_g4_sanity"
# evaluate on validation + corruptions_val ONLY -- never the frozen test benchmark
python scripts/evaluate_calibration.py --run "$OUTPUT_ROOT/validation/C2_g4_sanity" \
    --corruption-root "$OUTPUT_ROOT/corruptions_val"
```

G4: PASS if clean validation top-1 ≥ 0.95, FAIL below.

**Pre-registered attribution, recorded before the run.** If G4 fails, §2.3 and §3.2
already identify the two candidate causes, and they are separable: a failure
concentrated in the small-lesion and texture-dominated classes (Septoria, Cedar apple
rust, powdery mildews) implicates the **7×7 spatial bottleneck**; a uniform failure
across all classes implicates the decoder's capacity to render a classifiable image
at all. The per-class breakdown from `evaluate_calibration.py` distinguishes them.

---

## 6. Manuscript change map for C2

Separated as required. **Category C is not described as clarification.**

### A — Clarification of previously unspecified implementation details

The manuscript never states these; supplying them adds information without changing
any claim.

| Element | Change |
|---|---|
| §3.2 | State the encoder's depth, width, head count and patch/stride for the first time. `D`, `L`, `N`, `P` are symbolic in every equation and no variant, size or parameter count appears anywhere in the submitted text. |
| §3.3 | State the AE's latent dimension, channel widths and activation — never specified. |
| §3.1 | State the PE formulation (1-D vs 2-D sinusoidal) and γ — never specified. |
| §4.2 | State the AE loss weighting and warm-up schedule — never specified. |

### B — Correction of internal manuscript inconsistencies

The manuscript contradicts itself or is provably wrong; these correct it.

| Element | Inconsistency | Correction |
|---|---|---|
| §5.1 / §5.3 vs §3.3 / Fig. 2 | §5.1 says the AE "operates on intermediate feature representations rather than raw images" and §5.3 says fusion works on "fixed-dimensional latent features rather than directly on image pixels" — but §3.3 and Fig. 2 depict an image-space AE | Adopting a feature-space AE makes §5.1 and §5.3 true; they need no edit once the architecture matches them |
| §5.1 | "this overhead is modest compared to the backbone feature extractor" — false of the submitted architecture (88× the classifier's FLOPs) | Replace with the measured table |
| Eq. (3) | Both lines are printed identically, so the MLP and attention sub-blocks are indistinguishable | Re-typeset as two distinct residual sub-blocks |
| Eq. (5) | Programming-style assignment, same symbol both sides | `Δ_sparse(W,b) = Δ(W,b) + β Σⱼ KL(ρ‖ρ̂ⱼ)` |
| Eq. (6) | Summation index printed `1=1`; stray parentheses after `y⁽ⁱ⁾` | Re-typeset; define `m`, `n_l`, `s_l`, `λ` |
| §3.3 prose | "x is normal RGB input, while x̂ is an input image with noise adding" inverts the denoising convention | Corrupted input, clean reconstruction target |
| §4.2.2 / §4.2.3 | Random initialisation, 50 epochs, lr 0.01, momentum 0.937 — all contradicted by the training log | Report the resolved values |

### C — Genuine methodological changes introduced during Major Revision

**These are changes to the method, not clarifications, and must be disclosed as such
in the response letter.**

| Element | Change | Why it is category C |
|---|---|---|
| **Encoder identity** | ViT-B/16 → MobileViT-XXS | The manuscript never named a size, but the historical code referenced a ViT-B/16 checkpoint. Reviewers must not be left to infer the paper always used a 0.95 M encoder. |
| **Eq. (2)** | Must be **replaced**, not merely re-parameterised | Eq. (2) defines a ViT patch embedding producing `[i_class; i_p⁽¹⁾E; …] + E_pos`, an `(N+1, D)` token sequence with a CLS token. MobileViT emits spatial grids (`[B,16,112,112] … [B,320,7,7]`) with **no CLS token and no flat token sequence**. Eq. (2) is false of C2 as written. |
| **Eq. (4)** | Retained for C2 | MobileViT performs standard spatial self-attention on unfolded patches, so the softmax MHA formulation still holds. *(C3 would invalidate it — EfficientViT uses linear attention.)* |
| **AE operating space** | Image space (224²) → feature space (7×7) | Fig. 2 and §3.3 depict image-space fusion. Aligning with §5.1/§5.3 resolves a contradiction, but it still changes what the method does. |
| **Definition of TF-RGB** | The 3-channel image-space projection → the raw backbone grid | §3.3 says "TF-RGB and PE-RGB are input to the sparse autoencoder". In C2 the AE consumes the raw `[B,320,7,7]` map and a pooled PE map — different objects. |
| **PE-RGB's role** | Full-resolution branch → 7×7 pooled, 0.040% of fused magnitude | §3.2/§3.3 |
| **Fig. 2** | Redraw: replace the flattened-patch column with the encoder's stage diagram; move the fusion ⊕ and AE to the 7×7 grid; **show the PE average-pool**; annotate latent as `[64,7,7]`; annotate all component sizes | The figure currently depicts neither the encoder nor the fusion location accurately |
| **Complexity discussion** | Replace §5.1's qualitative paragraph and re-ground §5.2's deployment claim in measured numbers | Currently asserted, not measured |

---

## 7. PASS / FAIL recommendation for C2

### **CONDITIONAL FAIL — do not adopt C2 as currently specified.**

Not because it is expensive; it comfortably meets the cost targets on the criteria
measurable so far (1.71× params, 2.74× FLOPs). It fails on **what the efficiency
costs**:

1. **The 7×7 grid removes the discriminative signal.** Each cell covers 32×32 px
   while target lesions are 5–15 px. The qualitative grid shows small spots, lesion
   boundaries and texture erased, while healthy-leaf appearance survives — the loss
   is concentrated exactly where the class information is. This is geometric, not
   fixable by adding channels.
2. **C2 deletes the PE branch.** At 0.040% of fused magnitude and swamped 243:1 by
   unnormalised backbone activations, PE-RGB is not a component of C2 in any
   meaningful sense. The paper would be claiming a three-component method while
   deploying two.
3. **The decisive test has not been run**, and cannot be until a CUDA device is
   available (C2-1 → C2-3).

The result is not "C2 is wrong" but "C2 is not yet validated, and two specific,
measured mechanisms predict it will fail G4." Adopting it now would be premature.

### Recommended next step, in order

1. **Run C2-1 on the T4** (§1). Cheap, decisive for the latency rule, unblocks C2-3.
2. **Run C2-3 / G4** (§5) regardless of the concerns above — it is one short run and
   it converts a prediction into a measurement. The pre-registered attribution in §5
   makes the outcome interpretable either way.
3. **If G4 fails**, do not tune C2. Two documented alternatives already exist and
   neither is a post-hoc rescue:
   - **C2-14**: take the stride-16 stage (`[B,64,14,14]`) instead of stride-32.
     Latent `[64,14,14]` = 12,544 dims, compression 12× instead of 48×, cells of
     16×16 px instead of 32×32. Costs FLOPs but stays far below C1.
   - **Option B** (training-only auxiliary branches) from the redesign dossier, whose
     trigger condition — "the AE adds nothing at inference" — would be partly met.

No re-freeze is justified until at least steps 1 and 2 are complete.

---

## 8. Unresolved scientific risks

| # | Risk | Status |
|---|---|---|
| R1 | 7×7 spatial bottleneck destroys lesion-scale detail | **Measured and confirmed** at the spatial floor; effect on end classification unmeasured |
| R2 | Decoder cannot render a classifiable image from 3,136 latent values | **Untestable without training** — the core open question |
| R3 | C2's fusion swamps PE 243:1 | **Measured and confirmed**; fixing it requires a fusion-semantics change |
| R4 | PE carries no image-dependent information in any candidate | **Measured** (99.7% constant); ablation `A1_pe_only` will confirm independently |
| R5 | C2 latency may exceed 3× baseline despite passing on FLOPs | **Unmeasured** — MobileViT is bandwidth-bound on T4 |
| R6 | Decode-then-classify is architecturally redundant | **Unexamined.** C2 renders semantically rich features back into a blurry image so a CNN can re-extract features from it. This round trip cannot add information and is the strongest structural argument for Option B. |
| R7 | Separability probes are weak in absolute terms | **Acknowledged** — supports only relative conclusions (§2.4) |
| R8 | Eq. (2) must be replaced, contradicting the redesign dossier's earlier claim that Option C needed no equation changes | **Corrected here.** MobileViT produces no CLS token or flat token sequence, so Eq. (2) is false of C2. |

---

## Appendix — reproducing

```bash
python scripts/audit_bottleneck.py --per-class 12          # this audit, ~468 images
python scripts/benchmark_architectures.py --device cuda --pretrained   # C2-1, needs T4
python scripts/check_shapes.py --device cpu                # frozen v1 unaffected: 16/16
```

Artifacts: `bottleneck_audit.json`, `spatial_floor_grid.png` under
`${OUTPUT_ROOT}/architecture_v2/bottleneck_audit/`.
