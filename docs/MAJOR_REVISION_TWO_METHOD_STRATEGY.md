# Major Revision — Two-Method Research Design

**Date:** 16 August 2026 · **Status:** documentation + matrix proposal. Nothing trained, nothing re-frozen, no manuscript edited.
**Agreed with the supervising professor:** the revision presents **two** methods, not a replacement.

Evidence labels used throughout — **[MEASURED]** on real data/code · **[RECOVERED]** from a surviving
historical artifact · **[RECONSTRUCTED]** rebuilt because the original is unavailable · **[DERIVED]**
arithmetic from measurements · **[HYPOTHESIS]** reasoned but unverified · **[NOT YET TESTED]**.

> **Update 16 Aug 2026 — the T4 benchmark has been executed (§3).** Five defects
> found afterwards are recorded in **`PROTOCOL_AMENDMENT_2026-08-16.md`**, with the
> decoder trace in **`DECODER_PATH_AUDIT.md`**. Three are now **fixed in code**:
> the FLOPs handler, the dead projection-head parameters, and the order-dependent
> latency criterion. Every FLOPs and parameter figure in this document is the
> corrected one; the raw benchmark is preserved unedited at
> `docs/evidence/2026-08-16_t4_benchmark_raw.json`.
> **C2-28 is a CONDITIONAL CANDIDATE, not yet validated, and not to be called
> "efficient" on cost evidence alone** (§3.3).

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
**[MEASURED]** 87,549,123 params · **34.8731 GFLOPs** · 58.83× baseline params · **84.73× baseline FLOPs**
(corrected accounting; 36.2215 / 88.00× under the defective handler).

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
| "this overhead is modest compared to the backbone feature extractor" | §5.1 | **[MEASURED] refuted** — 84.73× the classifier's FLOPs |
| "the auto-encoder operates on intermediate feature representations rather than raw images" | §5.1 | **[MEASURED] false** of the reconstruction (image-space AE) |
| "the fusion mechanism operates on fixed-dimensional latent features rather than directly on image pixels" | §5.3 | **[MEASURED] false** of the reconstruction |
| "real-time inference on resource-constrained devices" | §5.2 | **[MEASURED]** not supportable: 34.87 GFLOPs, **28.51 ms/image on a T4** (8.48× the classifier) |
| Model "not initialized with pre-trained weights … randomly initialized" | §4.2.2 | **[RECOVERED] contradicted** — `pretrained=True`, 156/158 transferred |
| 50 epochs / 4.017 h; lr 0.01; momentum 0.937 | §4.2.3 | **[RECOVERED] contradicted** — 30 epochs / 1.980 h; lr 7.14e-4 resolved by `optimizer=auto` |
| "61,486 images" | Abstract, §4.1 | **[RECOVERED] contradicted** — 55,259 |
| Blueberry greenhouse results | §4.6.4, Figs. 11–12 | **[MEASURED]** only 74 images across 8 usable classes — withdrawn |
| §4.6.2's "proposed + YOLOv7 = 14.8%" | §4.6.2 | **[MEASURED]** does not reproduce (archived runs give 16.40% / 16.55%) |
| Top-5 in Tables 2–3 | §4.6 | **[DERIVED]** undefined under the recovered `max_det=1` protocol |
| "Our research is the first to introduce…" | §3 | unqualified priority claim |

### 1.2 EFFICIENT AE-TFPE

Leading candidate **C2-28**. **[MEASURED]** **1,716,586** params · **1.0126 GFLOPs** · 1.15× baseline params · **2.46× baseline FLOPs** · **10.2045 ms** batch-1 on a T4 (3.034×).
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

## 3. Task 3 — T4 benchmark · **EXECUTED 16 Aug 2026**

Tesla T4, all five models in one process on one GPU, batch 1 and 32, 50 warm-up,
200 timed iterations, one AMP policy. Full results and the two commissioned audits
are in `ARCHITECTURE_V2_BENCHMARK.md` §8 and `PROTOCOL_AMENDMENT_2026-08-16.md`.

| Model | Params | ×base | GFLOPs *(corrected)* | ×base | **BS1 latency** | ×base | BS32/img | ×base |
|---|---|---|---|---|---|---|---|---|
| YOLOv8n-cls baseline | 1,488,247 | 1.00× | 0.4116 | 1.00× | **3.3632 ms** | 1.000 | 0.22906 ms | 1.000 |
| **Original AE-TFPE (C0)** | 87,549,123 | 58.83× | 34.8731 | 84.73× | 28.5088 ms | 8.477 | 10.7734 ms | 47.033 |
| C2-7 | 2,544,634 | 1.71× | 0.9789 | 2.378× | 27.0000 ms | 8.028 | 1.2318 ms | 5.378 |
| C2-14 | 2,066,890 | 1.39× | 1.0046 | 2.441× | 16.3708 ms | 4.868 | 1.2260 ms | 5.352 |
| **C2-28 (leading Efficient candidate)** | **1,716,586** | **1.15×** | 1.0126 | 2.460× | **10.2045 ms** | **3.034** | 1.2908 ms | 5.635 |

GFLOPs and parameters are the **corrected** values, produced by the fixed
measurement tool. The raw JSON figures (88.00× / 2.74× / 3.38× / 4.43×, and
parameter counts 969 / 201 / 153 higher for the C2 family) are preserved in
`docs/evidence/2026-08-16_t4_benchmark_raw.json` and tabulated side by side in
`ARCHITECTURE_V2_BENCHMARK.md` §8. Both defects are fixed in code; **no output
changed**. Peak CUDA memory is in §3.1/§3.2.

⚠ **Batch-1 latency carries 15–30 % relative standard deviation.** C2-28 is
3.034× on the mean and **2.923× on the median** — the two straddle the 3×
threshold, and the rule does not say which statistic applies (amendment §A5).
Batch-32 timings are tight (0.9–2.7 %).

**FLOPs mis-rank these architectures.** C2-7 has the fewest FLOPs and is 2.6×
slower than C2-28 at batch 1. Selection on FLOPs alone would have picked the
slowest viable candidate.

### 3.1 Original AE-TFPE → Efficient AE-TFPE (C2-28) — **[DERIVED]** from the T4 run

The two-method comparison the revision is built on:

| Quantity | Original AE-TFPE | Efficient AE-TFPE (C2-28) | Reduction | Factor |
|---|---|---|---|---|
| Parameters | 87,549,123 | 1,716,586 | **−98.04 %** | **51.0×** |
| GFLOPs *(corrected)* | 34.8731 | 1.0126 | **−97.10 %** | **34.4×** |
| GFLOPs *(as reported)* | 36.2215 | 1.8215 | −94.97 % | 19.9× |
| Model size (fp32) | 334.576 MB | 7.156 MB | **−97.86 %** | **46.8×** |
| **BS1 latency** | 28.5088 ms | 10.2045 ms | **−64.21 %** | **2.79×** |
| BS32 per-image | 10.7734 ms | 1.2908 ms | −88.02 % | 8.35× |
| BS32 throughput | 92.82 img/s | 774.78 img/s | **+734.7 %** | **8.35×** |
| Peak CUDA memory, BS1 | 378.12 MiB | 65.21 MiB | **−82.75 %** | **5.80×** |
| Peak CUDA memory, BS32 | 646.25 MiB | 287.75 MiB | **−55.47 %** | **2.25×** |

### 3.2 Efficient AE-TFPE (C2-28) versus YOLOv8n-cls — the overhead that remains

| Quantity | YOLOv8n-cls | C2-28 | Overhead |
|---|---|---|---|
| Parameters | 1,488,247 | 1,716,586 | **+15.4 %** (1.153×) |
| GFLOPs *(corrected)* | 0.4116 | 1.0126 | **+146.0 %** (2.460×) |
| GFLOPs *(as reported)* | 0.4116 | 1.8215 | +342.5 % (4.425×) |
| Model size (fp32) | 5.703 MB | 7.156 MB | +25.5 % (1.255×) |
| **BS1 latency** | 3.3632 ms | 10.2045 ms | **+203.4 %** (3.034×) |
| BS32 per-image | 0.22906 ms | 1.2908 ms | **+463.5 %** (5.635×) |
| BS32 throughput | 4365.58 img/s | 774.78 img/s | **−82.3 %** |
| Peak CUDA memory, BS1 | 41.61 MiB | 65.21 MiB | **+56.7 %** (1.567×) |
| Peak CUDA memory, BS32 | 130.61 MiB | 287.75 MiB | **+120.3 %** (2.203×) |

### 3.3 Required interpretation — binding on the manuscript

> **Efficient AE-TFPE substantially reduces the computational burden of Original
> AE-TFPE, but still incurs meaningful overhead over YOLOv8n-cls.**

**Forbidden phrasings:** "negligible overhead", "baseline-equivalent efficiency",
or any wording implying C2-28 is free relative to the classifier. Both are
contradicted by 3.034× BS1 latency and 5.635× BS32 per-image cost.

C2-28 is a **CONDITIONAL CANDIDATE, not yet validated**: parameters preferred,
FLOPs conditional on amendment §A2, BS1 latency conditional and only just above
the 3× band, BS32 throughput a substantial overhead. **No C2-28 model has been
trained**, so nothing is yet known about its accuracy or robustness. It must not
be called "efficient" on the strength of a cost benchmark alone.

### 3.4 Latency bands

Bands apply to **Efficient AE-TFPE only** (Original AE-TFPE's cost is a finding
under RQ4, not a disqualification): ≤3× preferred · 3–5× acceptable only with
demonstrated robustness/information benefit · >5× requires reconsideration.

At **3.034× BS1**, C2-28 sits in the *acceptable-only-with-demonstrated-benefit*
band by a margin of 0.034×. That benefit is exactly what remains unproven, and
what E5/E3 must establish. Under the pre-amendment reading — BS32 as the
criterion — C2-28 would sit at 5.635× and fall in the *requires reconsideration*
band; see amendment §A1.5, where this is stated openly as the amendment's effect.

---

## 4. Task 4 — C2-28 clean sanity experiment · **PREPARED, NOT EXECUTED**

Cleared to run: both post-benchmark audits found **no defect that changes what
C2-28 computes**. The four items in `PROTOCOL_AMENDMENT_2026-08-16.md` concern a
decision rule, a FLOPs convention, a metadata string and 153 never-executed
parameters — none of which affects whether the 28×28 representation is learnable,
which is the only question this run asks. Execution awaits your go-ahead.

```bash
# COLAB -- ONE run. Validation only. No test, no corruptions_test.
python scripts/train.py --config configs/aetfpe_full.yaml \
    --override model.tf_backbone=mobilevit_xxs \
    --override model.ae_space=feature \
    --override model.tf_stage=2 \
    --out "$OUTPUT_ROOT/validation/C2_28_clean_sanity"
```

Clean validation only, so **no corruption set is required** and none is generated.
The `evaluate_calibration.py` step previously listed here is **not** part of this
run; it needs `corruptions_val`, which is out of scope.

**Objective, pre-registered:** *can C2-28 learn the 39-class problem without
catastrophic information loss?* Not superiority. Not adoption.

**Recorded automatically:** per-epoch training curve, validation top-1 **and
top-5**, convergence behaviour, AE reconstruction and KL loss, wall-clock, device,
and — since commit `da9d1ba` — **peak CUDA allocated/reserved memory**, reset
immediately before the first training step and written to `train_summary.json`
under `peak_memory`. The note that `train.py` does not log peak memory is
superseded.

**Verdict bands, pre-registered and not to be revised after seeing the result:**
≥ 0.95 learnable · 0.80–0.95 degraded, report and do **not** tune · < 0.80
catastrophic, stop and report. Diagnostic attribution: a flat or rising AE
reconstruction loss implicates the decoder; a falling one with stagnant accuracy
implicates the grid/interface. Full readout cells in `COLAB_EXECUTION_PACKAGE.md`.

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

## 7. Task 7 — Revised experiment matrix (**candidate minimum, NOT final**)

> **CORRECTION, 16 Aug 2026, per supervisor review.** The 19-run matrix below is
> the **current minimum candidate matrix**, not the final frozen matrix.
> **RQ2/RQ3 conclusions measured on Original AE-TFPE are NOT assumed to transfer
> to Efficient AE-TFPE**, because Efficient AE-TFPE changes *both* the transformer
> encoder *and* the AE operating space — two simultaneous changes, either of which
> could alter which components matter and which fusion operator wins. Whether an
> additional Efficient-side fusion control is required will be decided **after E5
> and E3 evidence exists**, not now.


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
- **Efficient-side fusion comparison** (add / concat / attention at stage 2): **DEFERRED, not excluded.** RQ3 is answered on Original AE-TFPE first. Whether the operator ranking transfers is an open question — Efficient AE-TFPE changes both the encoder and the AE operating space — and is to be decided **once E5 and E3 results exist**. It is explicitly *not* assumed to transfer.
- **Efficient-side full component ablation** (E1, E2, E4): **DEFERRED on the same basis.** E3 provides the first evidence on whether the AE conclusion transfers; the remaining component arms are revisited after that evidence, not ruled out in advance.

### 7.3 Totals

| | Count |
|---|---|
| Unique trainings, frozen v1 | 16 |
| New Efficient runs | +3 |
| **Total unique trainings (candidate minimum)** | **19** |
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

## 9a. Newly identified risk — determinism

**[MEASURED]** while instrumenting the trainer, using the **baseline arm only**
(pure frozen-v1 code paths):

| Device | Two identical runs, seed 0 | Result |
|---|---|---|
| CPU | `train_loss` 3.73670347 vs 3.73670347 | **identical** |
| MPS | `train_loss` 3.54054185 vs 3.73592884 | **divergent** |
| CUDA | — | **[NOT YET TESTED]** |

`SCIENTIFIC_PROTOCOL_FROZEN.md` asserts `seed=0, deterministic=True`. That holds on
CPU, fails on Apple MPS, and is **unverified on the actual training device**. No
current conclusion rests on an MPS number — every MPS run to date was a labelled
plumbing check — but this must be resolved before `revision-protocol-v2`.

`seed_everything()` sets `cudnn.deterministic=True` but not
`torch.use_deterministic_algorithms(True)`. Strengthening it would change training
behaviour, so it has **not** been changed; the Colab package tests CUDA determinism
in ~2 minutes first, so the decision can be made with evidence.

---

## 10. What is deliberately not done

No training. No re-freeze. No G5. No ablation campaign. No corruption test-set
generation. No test-set access. No manuscript source edited. No equations rewritten.
No figure redrawn. No method renamed. No PE modification. No frozen config altered.
Original AE-TFPE (C0) retained in full.

**Awaiting approval before proceeding.**
