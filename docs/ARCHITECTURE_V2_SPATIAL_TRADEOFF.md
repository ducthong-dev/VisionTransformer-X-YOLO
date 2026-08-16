# Architecture v2 — Spatial Resolution Trade-off

**Date:** 16 August 2026 · **Status:** measurement only. Nothing trained, nothing re-frozen.
**Frozen v1 untouched:** `check_shapes.py` 16/16, A5/D1 still 87,549,123 params, no `configs/` file modified.

Evidence labels used throughout: **[MEASURED]** on this machine · **[DERIVED]** arithmetic from
measurements · **[HYPOTHESIS]** reasoned but unverified · **[NOT YET TESTED]** requires a run
that has not happened.

---

## 0. Summary

**Recommendation: C2-28, conditionally.** It is the only candidate whose spatial
grid resolves lesion-scale structure, it has the *fewest* parameters of the three
(1.15× baseline), and it does not numerically annihilate the PE branch. It sits in
the **Conditional** band on FLOPs (4.43×), which the pre-registered rules permit
only with a clear scientific justification — which exists and is stated in §7.

Two findings changed the picture from the previous report:

1. **Higher spatial resolution makes the backbone *cheaper*, not more expensive.**
   Taking an earlier MobileViT stage truncates the network. Parameters fall
   monotonically from C2-7 → C2-28 while FLOPs rise, because cost migrates from
   backbone to decoder. This was not anticipated.
2. **The "C2 deletes the PE branch" finding is specific to C2-7 and does not
   generalise.** The 283× scale mismatch is an artefact of stage 4's 320-channel
   post-expansion layer. At stage 3 it is 23×, at stage 2 it is 11×. Reported as a
   correction; PE itself was not touched.

---

## 1. Stage 1 — C2 CUDA benchmark · **NOT EXECUTED**

This machine has no CUDA device (`torch.cuda.is_available() == False`), and your
standing environment constraint reserves training and all runtime measurement for
the Colab T4. Per *"Do NOT estimate unavailable values"*, latency, throughput and
peak CUDA memory are **absent from this document rather than estimated**.

The harness implements your exact protocol (batch 1 and 32, 50 warm-up, 200 timed,
`torch.cuda.synchronize()`, identical AMP):

```bash
# COLAB / Tesla T4 -- includes all four candidates plus the baseline
python scripts/benchmark_architectures.py --device cuda --pretrained \
    --only BASELINE C2 C2-14 C2-28 C3
```

**[MEASURED]** and hardware-independent, so already final:

| | Total params | Trainable | GFLOPs |
|---|---|---|---|
| BASELINE | 1,488,247 | 1,488,247 | 0.4116 |
| C2-7 | 2,545,603 | 1,594,579 | 1.1279 |
| C2-14 | 2,067,091 | 1,575,747 | 1.3896 |
| C2-28 | 1,716,739 | 1,567,219 | 1.8215 |
| C3 | 2,264,323 | 1,581,715 | 0.8055 |

**[NOT YET TESTED]** latency b=1, latency b=32, throughput, peak CUDA memory.

**[HYPOTHESIS]** MobileViT is memory-bandwidth-bound and historically
under-performs its FLOP count on a T4, so a candidate passing on FLOPs may still
land a band higher on latency. C2-28 at 4.43× FLOPs is the most exposed to this.

---

## 2. Stage 2 — C2-7 G4 diagnostic · **NOT EXECUTED**

G4 requires training on the full 38,584-image train split. Your standing
environment constraint is explicit: *"Before any command expected to train for many
epochs, process the full dataset ... STOP and provide the exact Google Colab
command/script instead."* **[DERIVED]** a local MPS run would take roughly 2.3 h,
which is squarely in that category.

```bash
# COLAB -- one diagnostic run, validation only, never test or corruptions_test
python scripts/train.py --config configs/aetfpe_full.yaml \
    --override model.tf_backbone=mobilevit_xxs \
    --override model.ae_space=feature \
    --out "$OUTPUT_ROOT/validation/C2_7_g4_sanity"

python scripts/evaluate_calibration.py \
    --run "$OUTPUT_ROOT/validation/C2_7_g4_sanity" \
    --corruption-root "$OUTPUT_ROOT/corruptions_val"
```

`evaluate_calibration.py` has no code path that can construct a test-split path,
and `select_v1.py`-style tripwires guard against test contamination.

**Interpretation is pre-registered and conservative.** A G4 PASS would show only
that the 7×7 representation retains enough information for *clean* classification.
It would not adopt C2-7, would not invalidate the lesion-loss evidence in §6, and
would say nothing about corruption robustness. A G4 FAIL strengthens the rejection.
No tuning either way.

---

## 3. C2-14 — exact architecture definition

**[MEASURED]** from the installed `timm` implementation of `mobilevit_xxs`.

| Property | Value |
|---|---|
| Source stage | `stages.3` — **genuine intermediate stage**, not an interpolation of 7×7 |
| Stage composition | `[BottleneckBlock, MobileVitBlock]` — **contains attention** |
| MobileViT block | depth 4, dim 80, patch size 2×2 |
| Output tensor | `[B, 64, 14, 14]` |
| Effective stride | 16 |
| Cell footprint | 16×16 input pixels |
| Backbone cost when truncated here | 491,344 params · 0.4598 GFLOPs |
| PE branch to grid | `adaptive_avg_pool2d(PE-RGB, 14×14)` → `[B, 3, 14, 14]` |
| AE input | concat → `[B, 67, 14, 14]` |
| AE encoder | Conv1×1 67→64, BN, Sigmoid — no spatial reduction |
| Latent | `[B, 64, 14, 14]` = **12,544 dims**, compression **12.0×** |
| Decoder | 4 stages, widths (48, 32, 16) then →3: 14→28→56→112→224 |
| Classifier interface | `[B, 3, 224, 224]` in [0,1] → **unmodified** YOLOv8n-cls |

**Receptive field.** `[HYPOTHESIS]` Each stage-3 position aggregates a 16×16 cell,
but the MobileViT block's unfolded-patch attention gives each position access to the
whole 14×14 map, so global context is present — it is spatial *localisation* below
16 px that is unavailable, not global reasoning.

---

## 4. C2-28 — exact architecture definition

| Property | Value |
|---|---|
| Source stage | `stages.2` — **genuine intermediate stage** |
| Stage composition | `[BottleneckBlock, MobileVitBlock]` — **contains attention** |
| MobileViT block | depth 2, dim 64, patch size 2×2 |
| Output tensor | `[B, 48, 28, 28]` |
| Effective stride | 8 |
| Cell footprint | 8×8 input pixels |
| Backbone cost when truncated here | 149,520 params · 0.3210 GFLOPs |
| PE branch to grid | `adaptive_avg_pool2d(PE-RGB, 28×28)` → `[B, 3, 28, 28]` |
| AE input | concat → `[B, 51, 28, 28]` |
| AE encoder | Conv1×1 51→64, BN, Sigmoid — no spatial reduction |
| Latent | `[B, 64, 28, 28]` = **50,176 dims**, compression **3.0×** |
| Decoder | 3 stages, widths (48, 32) then →3: 28→56→112→224 |
| Classifier interface | `[B, 3, 224, 224]` in [0,1] → **unmodified** YOLOv8n-cls |

**Controlled comparison.** Latent channel count (64), the decoder taper rule
(halving from 48), PE semantics, the fusion operator (concat), and the classifier
are all held fixed across C2-7 / C2-14 / C2-28. **Only the source stage changes**,
so the comparison isolates spatial resolution. No new attention mechanism was
added; no classifier was redesigned; PE was not modified.

**[MEASURED]** C2-28's latent grid (28×28) equals the latent grid of the rejected
C0/C1 (`StackedSparseDenoisingAE`, latent `[128,28,28]`) — at **1.8215 GFLOPs
versus C1's 3.0087**, i.e. the same spatial ceiling for 60% of the cost.

---

## 5. Complexity comparison · **[MEASURED]**

| Candidate | Params | ×base | Trainable | GFLOPs | ×base | FLOPs band |
|---|---|---|---|---|---|---|
| BASELINE | 1,488,247 | 1.00× | 1,488,247 | 0.4116 | 1.00× | — |
| **C2-7** | 2,545,603 | 1.71× | 1,594,579 | 1.1279 | **2.74×** | **Preferred** |
| **C2-14** | 2,067,091 | 1.39× | 1,575,747 | 1.3896 | **3.38×** | **Conditional** |
| **C2-28** | 1,716,739 | **1.15×** | 1,567,219 | 1.8215 | **4.43×** | **Conditional** |

None is hard-rejected (>5×).

### Where the cost sits · **[DERIVED]**

| Candidate | Backbone | AE + proj | YOLO | AE share of total |
|---|---|---|---|---|
| C2-7 | 0.5137 | 0.2026 | 0.4116 | 18% |
| C2-14 | 0.4598 | 0.5182 | 0.4116 | 37% |
| C2-28 | 0.3210 | 1.0889 | 0.4116 | **60%** |

**Parameters fall while FLOPs rise.** Selecting an earlier stage truncates the
backbone — `out_indices` prunes everything past it, so C2-28 never builds or runs
stage 3 or stage 4 — while the decoder must start at a higher resolution and
therefore costs more. Cost migrates from a parameter-heavy backbone to a
parameter-light but compute-heavy decoder. In C2-28 the decoder is 60% of total
FLOPs and the whole model is only 1.15× baseline parameters.

*(Observation, not an action: the decoder taper was fixed a priori and not tuned.
Its dominance in C2-28 is recorded for a future decision, not adjusted now.)*

---

## 6. Spatial-information comparison · **[MEASURED]**

Identical deterministic 468-image validation subset (12/class × 39 classes, sorted
selection, no randomness) for all three grids. Figure:
`results/architecture_v2/spatial_audit/spatial_floor_grid.png` — Original / 28×28 /
14×14 / 7×7 / |error|, same images throughout.

| Grid | Cell | Spatial compression | Latent @64ch | Latent compression | PSNR floor | SSIM floor |
|---|---|---|---|---|---|---|
| **7×7** | 32×32 px | 1024× | 3,136 | 48.0× | 20.04 dB | 0.383 |
| **14×14** | 16×16 px | 256× | 12,544 | 12.0× | 22.05 dB | 0.409 |
| **28×28** | 8×8 px | 64× | 50,176 | 3.0× | 24.09 dB | 0.462 |

These are **3-channel spatial floors** — a pessimistic reference, since the real
latent carries 64 channels. They are used for *relative* comparison across grids,
never as an approval criterion.

### Qualitative preservation · **[MEASURED]** from the figure

| Feature | 7×7 | 14×14 | 28×28 |
|---|---|---|---|
| Small spots (Septoria, Cedar apple rust) | **gone** | faint smudges, position ambiguous | **clearly visible** |
| Lesion boundaries (Late blight, Black rot) | **gone** | blurred but present | **sharp** |
| Vein-localized patterns | **gone** | **gone** | **visible** |
| Powdery-mildew texture | **gone** | **gone** | partially visible |
| Subtle discoloration | coarse tint only | present | **present** |
| Leaf silhouette / global colour | preserved | preserved | preserved |

### The geometric argument · **[DERIVED]**

Representative Septoria and rust lesions are ~5–15 px.

| Grid | Cell | Lesion vs cell |
|---|---|---|
| 7×7 | 32 px | lesion is **2–6× smaller than one cell** — position within a cell is unrecoverable |
| 14×14 | 16 px | lesion is **comparable to one cell** — presence encodable, localisation marginal |
| 28×28 | 8 px | lesion **spans one or more cells** — shape and position representable |

**28×28 is the first grid at which lesion structure is representable at all.** This
is geometry, not statistics, and it does not depend on channel count.

---

## 7. Pareto decision table

| Candidate | Grid | Cell | Params | ×P | GFLOPs | ×F | T4 latency ratio | PSNR floor | SSIM floor | Lesion preservation | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **C2-7** | 7×7 | 32 px | 2,545,603 | 1.71× | 1.1279 | **2.74×** *(Preferred)* | **[NOT YET TESTED]** | 20.04 | 0.383 | **None** — spots, boundaries, texture all lost | Cheapest; information-inadequate |
| **C2-14** | 14×14 | 16 px | 2,067,091 | 1.39× | 1.3896 | **3.38×** *(Conditional)* | **[NOT YET TESTED]** | 22.05 | 0.409 | **Marginal** — smudges, no localisation | Neither cheap nor sufficient |
| **C2-28** | 28×28 | 8 px | 1,716,739 | **1.15×** | 1.8215 | **4.43×** *(Conditional)* | **[NOT YET TESTED]** | 24.09 | 0.462 | **Yes** — spots, boundaries, veins | **Recommended, conditionally** |
| **Option B** | n/a | n/a | 1,488,247 | **1.00×** | 0.4116 | **1.00×** | 1.00× by construction | n/a | n/a | n/a — no inference-time AE | Fallback; strongest on cost, weakest on reviewer fit |

### Applying the pre-registered preference order

1. **"≤3× FLOPs AND materially improves lesion preservation over C2-7"** — **no
   candidate qualifies.** C2-7 is the only one ≤3×, and it is the reference it
   would have to improve on. Rule 1 yields nothing.
2. **"A 3–5× candidate may be retained only if its spatial preservation is
   substantially better and there is a clear scientific justification."**
   - **C2-28 qualifies.** +4.05 dB PSNR and +0.079 SSIM over C2-7; cell footprint
     4× finer; latent compression 16× less aggressive (48.0× → 3.0×); the only grid
     where lesions are representable. Scientific justification: the manuscript's
     central claim is robust *lesion-based* disease classification, and a
     representation that cannot resolve lesions cannot support it. It also has the
     fewest parameters of the three.
   - **C2-14 does not.** +2.01 dB and +0.026 SSIM is not "substantially better",
     and qualitatively spots remain unlocalised smudges. It occupies the worst
     position on the frontier: more expensive than C2-7, still information-poor.
3. **">5× rejected"** — none.
4. **Option B** — not triggered, since C2-28 is defensible under rule 2.

**Result: C2-28.**

### Conditions attached · all **[NOT YET TESTED]**

C2-28 is recommended *conditionally*, and the conditions are pre-registered:

- **T4 latency ≤5×**, or it is hard-rejected. **[HYPOTHESIS]** at 4.43× FLOPs and
  bandwidth-bound blocks, C2-28 is unlikely to reach the ≤3× Preferred band.
- **C2-7's G4 result**, the only accuracy datapoint that will exist. If C2-7 passes
  G4 comfortably, the case for paying 1.6× more FLOPs for C2-28 weakens on *clean*
  accuracy — though not on corruption robustness, which is the paper's actual claim
  and which G4 does not test.
- If C2-28 fails on latency, the fallback is **not** C2-14 (rule 2 excludes it) but
  **Option B**.

---

## 8. PE scientific-risk note

**PE was not modified, normalized, rescaled, projected, learned, or amplified.**
Everything below is measurement.

### Recorded risk

> The current fixed positional encoding may behave primarily as a deterministic
> spatial bias rather than an image-dependent information source.

**[MEASURED]** supporting evidence, unchanged from the previous audit: PE's effect
at the fusion input is **99.71% constant**, 0.29% image-dependent. PE perturbs the
transformer's features by 21.6% ± 16.8% relative, but with the same field for every
image.

### Correction to the previous report · **[MEASURED]**

The previous document stated that "C2 deletes the PE branch" on the basis of a 243×
scale mismatch. **That finding is specific to C2-7 and does not generalise.**
Measured across the three candidate stages on identical images:

| Candidate | Stage | Channels | F_TF RMS | F_PE RMS | Ratio | F_PE share of fused magnitude |
|---|---|---|---|---|---|---|
| C2-7 | 4 | 320 | 140.31 | 0.495 | **283×** | 0.040% |
| C2-14 | 3 | 64 | 11.58 | 0.502 | **23×** | 1.03% |
| C2-28 | 2 | 48 | 5.41 | 0.504 | **11×** | 2.47% |

The extreme mismatch is an artefact of stage 4's 320-channel post-expansion
activations, not a property of feature-space fusion. At stage 2 the PE branch
retains ~2.5% of fused magnitude — small, but not numerically annihilated.

This is diagnosis, not rescue: no normalization was introduced, and the stage
choice is driven by the spatial-information argument in §6, not by PE.

### What remains untested

**[NOT YET TESTED]** Whether PE contributes anything at all. That is the
pre-registered `A1_pe_only` ablation, which is **not run here**. If it returns a
null result, the PE contribution claim is demoted or removed, and the null is
reported. No post-hoc optimisation.

---

## 9. Provisional manuscript change map

**No manuscript edits made. No equations rewritten. Fig. 2 not redrawn. Method not
renamed.** Provisional only.

### A — Clarification of previously unspecified implementation details

| Element | Change |
|---|---|
| §3.2 | State the encoder's depth, width, heads and patch/stride — `D`, `L`, `N`, `P` are symbolic everywhere in the submitted text and no size is ever given |
| §3.3 | State the AE's latent dimension, channel widths and activation — never specified |
| §3.1 | State the PE formulation and γ — never specified |

### B — Correction of internal manuscript inconsistencies

| Element | Correction |
|---|---|
| §5.1 / §5.3 vs §3.3 / Fig. 2 | §5.1 ("intermediate feature representations rather than raw images") and §5.3 ("fixed-dimensional latent features rather than directly on image pixels") contradict §3.3 and Fig. 2. A feature-space AE makes §5.1/§5.3 true |
| §5.1 | "overhead is modest" — false of the submitted architecture; replace with measured figures |
| Eq. (3) | Both lines printed identically |
| Eq. (5) | Assignment notation, same symbol both sides |
| Eq. (6) | `1=1` summation index; stray parentheses |
| §3.3 prose | Denoising convention inverted |
| §4.2.2 / §4.2.3 | Initialisation, epochs, optimizer, LR contradicted by the training log |

### C — Genuine methodological changes introduced during Major Revision

**Not clarification. Must be disclosed as method changes in the response letter.**

| Element | Change |
|---|---|
| **Encoder identity** | ViT-B/16 → MobileViT-XXS. The manuscript never named a size, but the historical code referenced a ViT-B/16 checkpoint; reviewers must not infer the paper always used a 0.95 M encoder |
| **Eq. (2)** | **Must be replaced.** It defines a ViT patch embedding producing an `(N+1, D)` token sequence with a CLS token. MobileViT emits spatial grids and produces neither |
| **Eq. (4)** | **Retained** — MobileViT performs genuine spatial self-attention on unfolded patches, so the softmax MHA formulation still holds |
| **AE operating space** | Image space (224²) → feature space at a backbone stage |
| **Source stage** | If C2-28 is adopted: the AE consumes `stages.2` output (48 ch, stride 8), a specific intermediate representation the manuscript never describes |
| **Definition of TF-RGB** | The 3-channel image-space projection → the raw backbone feature grid |
| **PE-RGB's role** | Full-resolution branch → pooled to the backbone grid |
| **Fig. 2** | Redraw: encoder stage diagram instead of the flattened-patch column; fusion ⊕ and AE at the grid; show the PE pooling; annotate latent and all component sizes |
| **§5.1 / §5.2** | Replace assertions with measured complexity and re-ground the deployment claim |

---

## 10. Recommendation

### **C2-28 — conditionally, pending T4 latency and C2-7's G4 result.**

Not because it is cheapest — it is the most expensive of the three in FLOPs. Because
it is the only candidate that resolves the structures the paper is about, and it
does so at **1.15× baseline parameters**, the lowest of the three.

**Why not C2-7:** at a 32×32 px cell footprint, lesions 2–6× smaller than one cell
are unrepresentable. Cheapest and information-inadequate.

**Why not C2-14:** dominated. More expensive than C2-7 (3.38× vs 2.74×) while still
failing to localise lesions (+0.026 SSIM). It sits on no useful part of the frontier.

**Why not Option B yet:** rule 4 triggers only if no feature-space candidate is
defensible. C2-28 is defensible under rule 2. **[HYPOTHESIS]** Option B remains the
stronger answer to Reviewer #10.6 and the only one with zero overhead, and the
decode-then-classify round trip in every C2 variant — rendering semantically rich
features back into a blurry image so a CNN can re-extract features from it — remains
structurally redundant and cannot add information. That argument is unchanged by
this measurement and is the reason Option B stays live as the fallback.

### Decision tree, pre-registered

```
Run Stage 1 (T4) and Stage 2 (G4 on C2-7)
├─ C2-28 latency > 5x baseline ............... hard reject -> Option B
├─ C2-28 latency <= 5x
│  ├─ C2-7 G4 FAIL ........................... adopt C2-28 (7x7 shown inadequate)
│  └─ C2-7 G4 PASS ........................... C2-28 still preferred for the
│                                              robustness claim; decide with the
│                                              co-authors whether clean-accuracy
│                                              parity justifies 1.6x fewer FLOPs
└─ Any candidate needs tuning to pass ........ do not tune -> Option B
```

### Not done, deliberately

No training of C2-14 or C2-28. No G5. No ablation matrix. No corruption benchmark.
No test-set access. No `revision-protocol-v2`. No manuscript edits. No PE
modification. No frozen config altered.

---

## Appendix — reproducing

```bash
python scripts/benchmark_architectures.py --device cpu \
    --only BASELINE C2 C2-14 C2-28 C3 --out results/architecture_v2/spatial_sweep.json
python scripts/audit_bottleneck.py --per-class 12 --out-dir results/architecture_v2/spatial_audit
python scripts/check_shapes.py --device cpu          # frozen v1: expect 16/16
```

Tooling added for this investigation, inert under v1 defaults
(`tf_backbone=""`, `tf_stage=None`, `ae_space="image"`):

| File | Change |
|---|---|
| `src/aetfpe/features/timm_encoder.py` | `out_index` — stage selection with backbone truncation |
| `src/aetfpe/models/aetfpe.py` | `tf_stage` field; decoder taper auto-derived from the grid |
| `scripts/benchmark_architectures.py` | C2-14 and C2-28 candidates |
| `scripts/audit_bottleneck.py` | spatial floor swept over 7/14/28 on identical images; 5-panel figure |
