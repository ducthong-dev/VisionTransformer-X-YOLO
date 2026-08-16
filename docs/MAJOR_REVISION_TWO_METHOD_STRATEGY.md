# Major Revision — Two-Method Research Design

**Date:** 16 August 2026 · **Status:** documentation + matrix proposal. Nothing trained, nothing re-frozen, no manuscript edited.
**Agreed with the supervising professor:** the revision presents **two** methods, not a replacement.

Evidence labels used throughout — **[MEASURED]** on real data/code · **[RECOVERED]** from a surviving
historical artifact · **[RECONSTRUCTED]** rebuilt because the original is unavailable · **[DERIVED]**
arithmetic from measurements · **[HYPOTHESIS]** reasoned but unverified · **[NOT YET TESTED]**.

---

## 0. The revised scientific narrative

```
YOLOv8n-cls baseline
      ↓
Original AE-TFPE          the manuscript's proposed method, reconstructed
      ↓
computational / architectural analysis        <- Reviewer #10.6 prompted this
      ↓
Efficient AE-TFPE         NEW, introduced during Major Revision
      ↓
component ablation + fusion comparison
      ↓
robustness–efficiency analysis
```

**Original AE-TFPE is not deprecated.** It is the reference proposed architecture and
stays in the run matrix. Efficient AE-TFPE is an addition, presented explicitly as a
revision-time improvement — never retroactively as what the paper originally did.

---

## 1. Task 1 — Formal two-method distinction

### 1.1 ORIGINAL AE-TFPE

Corresponds to frozen arm `A5_aetfpe_full` (= candidate C0).
**[MEASURED]** 87,549,123 params · 36.2215 GFLOPs · 58.83× baseline params · 88.00× baseline FLOPs.

#### A. Statements directly supported by the original manuscript

| Statement | Location |
|---|---|
| The method has four steps: PE-RGB → transformer → TF-RGB → fusion → stacked AE → YOLOv8 | §3, Fig. 2 |
| PE is added to split image patches via sine/cosine encoding | §3.1, Eq. (1) |
| A stack of `L` transformer encoders with multi-head attention produces TF-RGB | §3.2, Eqs. (2)–(4) |
| TF-RGB and PE-RGB are fused and input to a sparse auto-encoder | §3.3 |
| The AE output is input to YOLOv8 **for training and predicting** | §3, §3.3, §3.4 |
| AE objective = squared error + β·KL sparsity + L2 | Eqs. (5)–(6) |
| 39 classes; 70/15/15 split; Top-1/Top-5 metrics | §4.1, §4.4, §4.5 |
| Three degradation types: pepper noise, 70% transparency, combined | §4.3 |

#### B. Implementation details **RECOVERED** from historical artifacts

| Detail | Evidence | Confidence |
|---|---|---|
| ViT checkpoint string `google/vit-base-patch16-224-in21k` | `feature_extractor_folder.py:10` | HIGH |
| **No transformer forward pass ever executed** — only `ViTImageProcessor` was called | `:3, :10, :13–16`; output reproduced byte-exactly by the preprocessor alone | HIGH |
| The historical transform was `0.2·LUT(x) + 0.8·x`, a fixed 256-entry pointwise LUT | MAE **0.0000**, exact byte match vs the surviving processed dataset; all other α ≥ 7.17 | HIGH |
| Classifier: YOLOv8n-cls, pretrained, **1,488,247** params, 156/158 transferred | `log-org-280223`; reproduced exactly | HIGH |
| Optimizer AdamW, lr 7.14e-4, seed 0, 30 epochs, batch 128, 224 px | `log-org-280223:2,26–27` | HIGH |
| Split 38,584 / 8,340 / 8,335 = 55,259; 39 classes = PlantVillage 38 + `Background_without_leaves` | log + on-disk dataset + archived `image_id` prefixes | HIGH |
| Evaluation was **detection-style**: `max_det=1, conf=0.25`, undetected images scored as errors | `validation.ipynb` cell 0; `calculate_top1.py` | HIGH |
| §4.6.2's YOLOv7 9.08% reproduces exactly (757 / 8,340) | archived `best_predictions.json` | HIGH |
| A second blend ratio (0.1/0.9) also existed | `validation.py` data path | MEDIUM |

#### C. Implementation details **RECONSTRUCTED** — original unavailable

**These are not recoveries. No code for any of them has ever been found.**

| Component | Reconstruction | Basis |
|---|---|---|
| PE-RGB | 2-D sinusoidal over a 16×16 patch grid, `d=3`, γ=0.1, added in image space | literal reading of Eq. (1) |
| Transformer branch | Frozen ViT-B/16, patch tokens (CLS dropped) → 1×1 conv → 3 ch → bilinear upsample | the recovered checkpoint *name* only |
| Auto-encoder | 3 conv encoder stages 224→112→56→28, latent `[128,28,28]`, 3 transposed decoder stages | Eqs. (5)–(6) intent |
| AE objective | MSE(x̂, x_clean) + β·KL(ρ‖ρ̂), β=1e-3, ρ=0.05; corrupted input, clean target | Eqs. (5)–(6), denoising convention corrected |
| AE loss weight / warm-up | 10.0 / 3 epochs | loss-magnitude ratio; classical stacked-AE pretraining |
| Fusion operators (add/concat/linear/attention) | as implemented in `fusion/ops.py` | Reviewer #12's requested comparators |
| Corruption Types 1–3 | salt-and-pepper at 6 ratios; α=0.7 overlay with a different-class distractor; noise-then-overlay | §4.3 prose; two internal contradictions resolved and documented |
| Training augmentation | hflip 0.5, RandAugment(2,9), RandomErasing 0.4 | log's `auto_augment=randaugment, erasing=0.4, fliplr=0.5` |

#### D. Claims **no longer supported** by current evidence

| Claim | Location | Status |
|---|---|---|
| "this overhead is modest compared to the backbone feature extractor" | §5.1 | **[MEASURED] refuted** — 88.00× the classifier's FLOPs |
| "the auto-encoder operates on intermediate feature representations rather than raw images" | §5.1 | **[MEASURED] false** of the reconstruction (image-space AE) |
| "the fusion mechanism operates on fixed-dimensional latent features rather than directly on image pixels" | §5.3 | **[MEASURED] false** of the reconstruction |
| "real-time inference on resource-constrained devices" | §5.2 | **[DERIVED]** not supportable at 36.22 GFLOPs |
| Model "not initialized with pre-trained weights … randomly initialized" | §4.2.2 | **[RECOVERED] contradicted** — `pretrained=True`, 156/158 transferred |
| 50 epochs / 4.017 h; lr 0.01; momentum 0.937 | §4.2.3 | **[RECOVERED] contradicted** — 30 epochs / 1.980 h; lr 7.14e-4 resolved by `optimizer=auto` |
| "61,486 images" | Abstract, §4.1 | **[RECOVERED] contradicted** — 55,259 |
| Blueberry greenhouse results | §4.6.4, Figs. 11–12 | **[MEASURED]** only 74 images across 8 usable classes — withdrawn |
| §4.6.2's "proposed + YOLOv7 = 14.8%" | §4.6.2 | **[MEASURED]** does not reproduce (archived runs give 16.40% / 16.55%) |
| Top-5 in Tables 2–3 | §4.6 | **[DERIVED]** undefined under the recovered `max_det=1` protocol |
| "Our research is the first to introduce…" | §3 | unqualified priority claim |

### 1.2 EFFICIENT AE-TFPE

Leading candidate **C2-28**. **[MEASURED]** 1,716,739 params · 1.8215 GFLOPs · 1.15× baseline params · 4.43× baseline FLOPs.
**[NOT YET TESTED]** — not adopted, not trained, no superiority claimed.

#### Inherited unchanged from Original AE-TFPE

- The three-component concept: positional information + global context + auto-encoder fusion
- Image-space classifier interface — YOLOv8n-cls consumes a `[B,3,224,224]` reconstruction, **unmodified**
- AE objective: sparse denoising (corrupted input, clean reconstruction target, KL sparsity, L2)
- Frozen, ImageNet-pretrained global-context encoder
- **PE formulation, untouched** — deliberately not modified (§5)
- Concatenation as the fusion operator feeding the AE

#### Changed, with motivation

| # | Change | From → To | Motivation |
|---|---|---|---|
| 1 | Global-context encoder | ViT-B/16 (85.8 M) → MobileViT-XXS (0.95 M) | **efficiency** |
| 2 | Encoder stage | final stage (7×7) → `stages.2` (28×28, stride 8) | **experimental** — spatial-information audit; also truncates the backbone |
| 3 | AE operating space | image space (224²) → feature space (28×28 grid) | **efficiency** + **manuscript-consistency** (makes §5.1/§5.3 true) |
| 4 | AE latent | `[128,28,28]` → `[64,28,28]` | **efficiency** |
| 5 | Decoder depth | 3 stages 28→224 (unchanged count, different entry) | **derived** from change 3 |
| 6 | PE spatial resolution at fusion | 224² → pooled to 28×28 | **consequence of change 3**, not an independent PE decision — see §5 |

**Change 6 is a side-effect, and is labelled as one.** The PE module itself is
byte-identical to Original AE-TFPE. Pooling arises because fusion now happens at
the grid. **[MEASURED]** at stage 2 the PE branch retains 2.47% of fused magnitude
(vs 0.040% at stage 4) — small, but not annihilated.

---

## 2. Task 2 — Research questions

No answer is assumed. Each RQ names the arms that decide it.

| RQ | Question | Decided by | Status |
|---|---|---|---|
| **RQ1** | Does Original AE-TFPE improve robustness over YOLOv8n-cls under controlled degradation? | A5 vs A0 on the frozen corruption benchmark | **[NOT YET TESTED]** |
| **RQ2** | What are the individual contributions of PE, TF, and the AE? | A0, A1, A2, A3, A4, A5 | **[NOT YET TESTED]** |
| **RQ3** | Does AE-based fusion outperform addition, concatenation, projection, attention? | F1, F2, F3(=A3), F4, F5(=D1) vs A5 | **[NOT YET TESTED]** |
| **RQ4** | What computational overhead does Original AE-TFPE introduce? | complexity benchmark | **[MEASURED]** — 58.83× params, 88.00× FLOPs; latency **[NOT YET TESTED]** |
| **RQ5** | Can Efficient AE-TFPE preserve the robustness benefit at substantially lower cost? | E5 vs A5 vs A0 on the same benchmark | **[NOT YET TESTED]** |
| **RQ6** | Does the AE actually produce a more degradation-resilient latent representation? | latent-drift + separability analysis on A5 and D1 | **[NOT YET TESTED]** |

RQ4 is the only one with a partial answer, and it is the one that motivated the
two-method design.

**A prior, recorded so it cannot be quietly dropped.** **[MEASURED]** the archived
YOLOv7 pair shows a 1.81× robustness gain produced by a zero-parameter lookup
table. RQ1 must therefore be answered *alongside* the mechanism controls
M1/M2/M3, or a positive RQ1 result cannot be attributed to the method.

---

## 3. Task 3 — T4 benchmark · **NOT EXECUTED**

`torch.cuda.is_available() == False` on this machine (macOS, arm64, torch 2.2.0
built without CUDA). Per *"Do not estimate missing CUDA measurements"*, latency,
throughput and peak GPU memory are **absent, not estimated**.

```bash
# COLAB / Tesla T4 -- all five models, one session, identical conditions
python scripts/benchmark_architectures.py --device cuda --pretrained \
    --only BASELINE C0 C2 C2-14 C2-28
```

Harness implements exactly: batch 1 and 32 · 50 warm-up · 200 timed ·
`torch.cuda.synchronize()` around every timed region · one AMP policy for all.
Writes `latency_ms_mean/std/median`, `throughput_img_per_s`, `peak_gpu_mem_mb`.

**[MEASURED]** and hardware-independent, already final:

| Model | Params | ×base | Trainable | GFLOPs | ×base |
|---|---|---|---|---|---|
| YOLOv8n-cls baseline | 1,488,247 | 1.00× | 1,488,247 | 0.4116 | 1.00× |
| **Original AE-TFPE (C0)** | 87,549,123 | 58.83× | 1,750,467 | 36.2215 | 88.00× |
| C2-7 | 2,545,603 | 1.71× | 1,594,579 | 1.1279 | 2.74× |
| C2-14 | 2,067,091 | 1.39× | 1,575,747 | 1.3896 | 3.38× |
| **C2-28 (leading Efficient candidate)** | 1,716,739 | **1.15×** | 1,567,219 | 1.8215 | 4.43× |

**[NOT YET TESTED]** latency b=1, latency b=32, throughput, peak GPU memory — for all five.

Latency bands apply to **Efficient AE-TFPE only** (Original AE-TFPE's cost is a
finding under RQ4, not a disqualification): ≤3× preferred · 3–5× acceptable only
with demonstrated robustness/information benefit · >5× requires reconsideration.

---

## 4. Task 4 — C2-28 clean sanity experiment · **NOT EXECUTED**

Requires full-split training; your standing environment constraint reserves that
for Colab. Pre-registered here in full so the run is interpretable whenever it happens.

```bash
# COLAB -- ONE run. Validation only. No test, no corruptions_test.
python scripts/train.py --config configs/aetfpe_full.yaml \
    --override model.tf_backbone=mobilevit_xxs \
    --override model.ae_space=feature \
    --override model.tf_stage=2 \
    --out "$OUTPUT_ROOT/validation/C2_28_clean_sanity"

python scripts/evaluate_calibration.py \
    --run "$OUTPUT_ROOT/validation/C2_28_clean_sanity" \
    --corruption-root "$OUTPUT_ROOT/corruptions_val"
```

**Objective, pre-registered:** *can C2-28 learn the 39-class problem without
catastrophic information loss?* Not superiority. Not adoption.

**Recorded automatically:** per-epoch training curve, validation top-1 and top-5,
convergence behaviour, AE reconstruction and KL loss, wall-clock, and device — all
already emitted by `train.py` into `metrics.csv` / `train_summary.json`.
Peak GPU memory is **not** currently logged by `train.py`; it is available from the
Task-3 benchmark instead.

**Interpretation, fixed in advance:**

| Outcome | Meaning |
|---|---|
| Validation top-1 comparable to A0 | 28×28 feature-space representation is learnable; proceed to the campaign proposal |
| Catastrophically poor | **STOP and report.** Do not tune. Report whether the 28×28 grid or the decode-to-image step is implicated, using the per-class breakdown |

**C2-7's G4 is no longer a gate on C2-28**, per this update. C2-7, C2-14 and C2-28
are independent architecture candidates; a C2-7 outcome neither blocks nor
validates C2-28.

---

## 5. Task 5 — Positional encoding: no rescue

**PE has not been modified, normalized, rescaled, amplified, projected or redesigned.**

**[MEASURED]** evidence on record:

| Quantity | Value |
|---|---|
| PE's effect at the fusion input that is constant across images | **99.71%** |
| Image-dependent fraction | **0.29%** |
| F_TF/F_PE per-element RMS ratio — stage 4 / stage 3 / stage 2 | 283× / 23× / 11× |
| F_PE share of fused magnitude — stage 4 / stage 3 / stage 2 | 0.040% / 1.03% / 2.47% |

The stage-dependence is a **measurement**, not a fix: the stage was chosen on the
spatial-information argument, and the milder PE mismatch at stage 2 is a
consequence, not a motivation.

**Recorded scientific risk:**

> The current fixed positional encoding may behave primarily as a deterministic
> spatial bias rather than an image-dependent information source.

**[NOT YET TESTED]** The pre-registered `A1_pe_only` ablation decides it, and is
**not run here**. If PE is null, the null is reported and the PE contribution claim
is demoted or removed — which is scientifically preferable to a post-hoc rescue.

---

## 6. Task 6 — Original AE-TFPE preserved

- `A5_aetfpe_full` (= C0) **remains in the run matrix**, unchanged and unrenamed.
- C2-28 is documented throughout as the **"leading Efficient AE-TFPE candidate"**,
  never as "the original architecture" or "the replacement architecture".
- The three-way comparison **YOLOv8n-cls vs Original AE-TFPE vs Efficient AE-TFPE**
  is now a core result, and is the direct answer to Reviewer #10.6.
- Earlier documents that labelled C0 "HARD REJECT" have been corrected: that verdict
  was reached under a *replacement-selection* rule set that no longer applies. C0 is
  the reference method; its cost is an RQ4 finding.

---

## 7. Task 7 — Revised experiment matrix (proposal only)

### 7.1 All 16 frozen runs remain necessary

| Group | Arms | Serves |
|---|---|---|
| Original ablation | A0, A1, A2, A3, A4, **A5 = Original AE-TFPE** | RQ1, RQ2 |
| Fusion comparison | F1, F2, F4, **D1** (F3 = A3, F5 = D1 by reuse) | RQ3 |
| Mechanism controls | M1, M2, M3 | RQ1 attribution — **shared by both methods**, no duplication |
| Fair baselines | B1, B2, B3 | RQ1 context, Reviewer #10.7 |

None is made redundant by the two-method framing. M1–M3 and B1–B3 are
architecture-independent and are reused by both methods.

### 7.2 New runs required — **3 only**

| ID | Configuration | Answers | Why not reusable |
|---|---|---|---|
| **E5** | Efficient AE-TFPE full = C2-28 | **RQ5** | The Efficient method itself |
| **E3** | Efficient PE+TF, **no AE** (stage 2, feature-space fusion, linear projection) | Does the AE still contribute *inside* the Efficient variant? | Original's A3 uses ViT-B/16 + image space — not transferable |
| **E7** | C2-7 (7×7 grid), otherwise identical to E5 | Spatial-resolution control — converts the design choice from an audit argument into a measured one | No existing arm varies only the grid |

**Deliberately excluded** — recorded so the omissions are visible, not accidental:

- **C2-14**: dominated on the measured frontier (costlier than C2-7, still cannot localise lesions). **[MEASURED]**
- **Efficient-side fusion comparison** (add / concat / attention at stage 2): RQ3 is answered once on Original AE-TFPE. **[HYPOTHESIS]** the ranking of fusion operators is a property of the concept, not the encoder. Mark **P2**; add only if a reviewer asks.
- **Efficient-side full component ablation** (E1, E2, E4): RQ2 answered on Original. E3 alone tests whether the AE conclusion transfers.

### 7.3 Totals

| | Count |
|---|---|
| Unique trainings, frozen v1 | 16 |
| New Efficient runs | +3 |
| **Total unique trainings** | **19** |
| Table rows served by reuse | F3 = A3, F5 = D1 |

**[DERIVED]** compute, extrapolated from the recovered 1.98 h / 30-epoch baseline
and measured per-arm FLOPs:

| Group | Runs | T4-hours |
|---|---|---|
| Plain-YOLO arms (A0, A1, M1, M2, M3) | 5 | ~10 |
| Image-space AE arm (A4) | 1 | ~2.2 |
| ViT-B/16 arms (A2, A3, A5, D1, F1, F2, F4) | 7 | ~17.5 |
| External baselines (B1, B2, B3) | 3 | ~13 |
| **Efficient arms (E5, E3, E7)** | **3** | **~5.5** |
| Corruption generation (val + test) | – | ~1.7 CPU |
| Final evaluation, 19 models × 26 configs = 494 | – | ~5 |
| Latent / complexity / figures | – | ~1.5 |
| **Total** | **19** | **≈ 56 T4-hours** |

Roughly a third on an A100/L4. **[DERIVED]** storage: checkpoints ≈ 2.8 GB
(ViT-B/16 arms are 335 MB each; Efficient arms ~10 MB), `corruptions_val` ≈ 2 GB,
frozen test benchmark ≈ 21 GB (**not yet generated**).

**Not launched.** This is a proposal.

---

## 8. Task 8 — Manuscript claim map

No manuscript edits made. "Expected revision" assumes the evidence lands supportive;
"Required revision" assumes it does not.

| # | Original claim | Evidence available now | Evidence still required | If supported | If unsupported |
|---|---|---|---|---|---|
| 1 | **Lightweight** / "overhead is modest" (§5.1) | **[MEASURED]** Original = 88.00× baseline FLOPs, 58.83× params | T4 latency (RQ4) | — | **Retract for Original.** Replace with the measured table. A reduced-overhead claim may be made **only** for Efficient AE-TFPE, and only after its own measurement |
| 2 | **Computational efficiency** (§5.1, §5.3) | **[MEASURED]** refuted for Original; §5.1/§5.3's "intermediate feature representations" describe an architecture the reconstruction does not implement | Efficient benchmark + sanity run | State Efficient's measured cost | Restrict all efficiency language to Efficient AE-TFPE; correct §5.1/§5.3 |
| 3 | **Noise-resilient latent representation** (Abstract, §1) | none | RQ6: latent drift + separability, A5 and D1 | Report drift ratio **with** separability, since drift alone is reducible by collapse | Remove the phrase; describe the AE as a reconstruction regulariser without a resilience claim |
| 4 | **Benefit of positional encoding** (Abstract, §1, §3.1) | **[MEASURED]** PE is 99.71% constant | RQ2 / `A1_pe_only` | Retain with the measured effect size | **Demote or remove.** Report the null. No post-hoc rescue |
| 5 | **Local/global feature fusion** (§1, §3) | architecture exists in both methods | RQ2 (A2, A3) | Retain | Reduce to a described component without a contribution claim |
| 6 | **AE fusion superior to conventional operators** (§1) | none | RQ3: F1, F2, F3, F4 vs D1/A5 | Retain with the comparison table | Retract the superiority claim; report AE fusion as *an* operator, not a better one |
| 7 | **Robustness under degradation** (Abstract "≈1.9×", Tables 2–3) | **[RECOVERED]** historical numbers are Protocol A, unreproducible; **[MEASURED]** a zero-parameter LUT alone gave 1.81× | RQ1 **plus** M1/M2/M3 | Retain, but attributed against the mechanism controls | If M1/M2/M3 match A5, reattribute the effect to the input transform and rewrite the contribution |
| 8 | **Real-world robustness** (§4.6.4, Figs. 11–12) | **[MEASURED]** only 74 blueberry images, 8 usable classes, one with a single image | none obtainable | — | **Withdraw §4.6.4 and Figs. 11–12**; state as a limitation; disclose in the response letter |
| 9 | **Deployment suitability** (§5.2) | **[MEASURED]** refuted for Original at 36.22 GFLOPs | Efficient T4 latency + sanity | Claim for Efficient only, with measured numbers | Remove the mobile/edge claim entirely |
| 10 | **Model-agnostic** (§1, §4.6.2) | **[MEASURED]** true for every arm except F2 (6-channel stem); **[RECOVERED]** the YOLOv6–v10 sweep is not reproducible | Either re-run variants (P2) or withdraw | Retain, with F2's stem modification disclosed | Withdraw the YOLO-variant generalisation; keep only "classifier is unmodified", which is measured |

**No claim is retained merely because it was in the submitted paper.**

---

## 9. Task 9 — Response-to-reviewer strategy (strategy only, not the letter)

### Reviewer #10

| Item | Strategy | Depends on |
|---|---|---|
| **10.1** distinction from prior transformer–AE fusion | New related-work positioning **plus** the RQ3 fusion table as empirical support, rather than a prose assertion | RQ3 |
| **10.3** why AE fusion over concat/attention | Direct answer from the fusion comparison; if AE does not win, say so | RQ3 |
| **10.4** "noise-resilient latent" not demonstrated | RQ6 latent-drift analysis, reported with separability so a collapse cannot masquerade as resilience | RQ6 |
| **10.5** comprehensive ablation | RQ2 six-arm component ablation on Original AE-TFPE, plus E3 to show whether the AE conclusion transfers | RQ2 |
| **10.6** quantitative computational overhead | **See below — the most important one** | RQ4, RQ5 |
| **10.7** were baselines retrained identically? | One frozen protocol for all 19 runs, published; B1–B3 retrained under it; the protocol table becomes an appendix | frozen protocol |
| **10.8** explicit limitations | New Limitations section: no field validation (blueberry withdrawn), reconstructed corruption definitions, historical results not reproducible, efficiency–robustness trade-off stated plainly | — |

#### Reviewer #10.6 — the framing

**Not:** "We clarified that AE-TFPE is lightweight." That would be false.

**Instead, once the experiments support it:**

> The reviewer's concern prompted us to quantify the computational overhead of the
> original formulation. We found it to be substantial — 58.8× the parameters and
> 88.0× the FLOPs of the YOLOv8n-cls baseline — and therefore introduce an
> efficiency-oriented variant, Efficient AE-TFPE, which we evaluate separately
> against the original formulation rather than in place of it.

**[NOT YET TESTED]** — this must not be written as a result until RQ4's latency and
RQ5 exist. The measured parameter and FLOP figures are already final; the robustness
half of the sentence is not.

### Reviewer #11

| Item | Strategy | Status |
|---|---|---|
| AE objective | Now fully specified: stacked **sparse denoising** AE, corrupted input → clean target, MSE + β·KL + L2, with warm-up. Labelled **RECONSTRUCTED** | ready to write |
| Fusion operator | Exact tensor shapes before/after fusion for both methods; RQ3 comparison | RQ3 |
| AE / classifier interface | Explicit: AE emits `[B,3,224,224]`; classifier **unmodified** in every arm but F2, which is disclosed | ready to write |
| Broader corruptions | 26-configuration frozen benchmark: 6 pepper ratios, transparency, 6 combined, plus Gaussian noise / blur / brightness / JPEG at 3 severities | benchmark defined, test set **not yet generated** |
| Ablation | RQ2 | pending |
| Compute overhead | RQ4 + the two-method comparison | partially measured |
| Deployment claims | Revised per claim map row 9 — restricted to Efficient, with measured numbers, or removed | pending |

### Reviewer #12

| Item | Strategy |
|---|---|
| Component ablation | A0 → A1 → A2 → A3 → A4 → A5, with `A1_pe_only` pre-registered as possibly null and reported either way |
| Fusion comparison | Addition, concatenation, concatenation+projection, attention, AE — all five requested comparators, with F2's stem modification disclosed |
| Terminology and equations | Eq. (3)/(5)/(6) defects, "multi-headed"→"multi-head", "GoogleLetNet"→"GoogLeNet", symbol definitions — all queued in the change map |

---

## 10. What is deliberately not done

No training. No re-freeze. No G5. No ablation campaign. No corruption test-set
generation. No test-set access. No manuscript source edited. No equations rewritten.
No figure redrawn. No method renamed. No PE modification. No frozen config altered.
Original AE-TFPE (C0) retained in full.

**Awaiting approval before proceeding.**
