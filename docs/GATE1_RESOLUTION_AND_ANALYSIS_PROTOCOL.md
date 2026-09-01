# GATE 1 Resolution, and the Test-Analysis Protocol Frozen Before Results

**Date:** 2 September 2026
**Written before any model was evaluated on any test distribution.**

## 0. What this document is, and what it is not

This is a **post-training analysis protocol**, frozen ahead of inspecting test results.

**It is NOT a pre-registration.** A pre-registration is written before the experiments it
governs. Training finished on 2026-09-01; this document was written on 2026-09-02. It
cannot retroactively acquire the epistemic standing of the gate it discusses, and no
part of it should ever be described as pre-registered.

The one genuinely pre-registered gate in this campaign is GATE 1, recorded in
`EXPERIMENT_CAMPAIGN_V2_PLAN.md` before any of these runs existed. **It was not passed.**
That fact is permanent and is not modified, relocated, or softened by anything below.

---

## 1. GATE 1 — the pre-registered gate, and its outcome

`EXPERIMENT_CAMPAIGN_V2_PLAN.md` §Stage B, committed before training:

> **GATE 1.** If C2-28 does not clearly exceed the best of M1/M2/M3 on clean validation,
> **stop the campaign**. Component ablations and fusion comparisons of a method that does
> not beat a gamma curve are not worth GPU time, and reporting that honestly is a
> stronger revision than burying it under nineteen runs.

Measured best validation top-1 **[MEASURED]**:

| Arm | Role | Best Val Top-1 |
|---|---|---|
| **E5 (C2-28)** | **the proposed method** | **0.9877786** |
| M1 | legacy-LUT control | 0.9958064 |
| M2 | photometric (gamma) control | 0.9976036 |
| M3 | augmentation control | 0.9968847 |
| A0 | plain YOLOv8n-cls baseline | 0.9976036 |
| B2 | EfficientNet-B0 external baseline | 0.9988018 |

**E5 did not exceed the best of M1/M2/M3. It did not exceed any of them, nor the plain
baseline. GATE 1 WAS NOT PASSED.**

E5 ranks 15th of the 16 completed arms. Only E7 (C2-7, 0.8436377) is lower.

### 1.1 No attempt is being made to redefine or erase the gate

The gate is not re-scoped to a different metric, a different split, a different arm set,
or a different threshold. It is not moved to the corruption axis. It is reported as a
pre-registered decision rule that returned a negative result on the quantity it named.

---

## 2. Two facts recorded alongside the outcome, not merged into it

### 2.1 Clean validation is strongly saturated

Fourteen of the sixteen completed arms fall within **0.9944884 – 0.9988018** — a spread
of **0.43 percentage points** across architectures that differ by 50× in parameter count.
At a single training seed, small ranking differences inside that cluster **should not be
interpreted as robust evidence of architectural superiority**, in either direction.

This does not rescue E5, and it is not offered as a defence of it.

### 2.2 E5's shortfall is larger than that cluster and is not hidden

E5 sits **0.98–1.10 percentage points below the strongest clean baselines**:

- 0.982 pp below A0 (0.9976036) and M2 (0.9976036);
- 1.102 pp below B2 (0.9988018).

That is roughly **2.3× the entire spread of the saturated group**. It is a reportable
measured difference, not noise, and it appears in the results tables and the abstract-level
summary rather than in a footnote.

---

## 3. Relationship between GATE 1 and the corruption evaluation

The robustness evaluation over Clean / Easy / Moderate / Hard **was already planned
independently**, as Stage E of `EXPERIMENT_CAMPAIGN_V2_PLAN.md`, before any of these
results existed. It is not a substitute instrument selected after seeing a bad number.

It is a **separate robustness analysis**. It answers a different question from GATE 1
and it does not, and cannot, overturn GATE 1's outcome.

> **Banned phrasing.** The following, and anything equivalent, must not appear in the
> manuscript, the response to reviewers, or any internal document:
> *"the gate was reformulated and passed on corruption"*;
> *"GATE 1 was superseded by the robustness results"*;
> *"clean accuracy is not the relevant criterion"* used as a reason to set the gate aside.
>
> The correct formulation is: **GATE 1 was not passed on clean validation. A separate,
> independently planned robustness analysis follows, and is reported on its own terms.**

Per §5 of `docs/CORRUPTION_PROTOCOL.md`, the corruption sets additionally **cannot
isolate noise resilience** — geometric, photometric, occlusion and noise effects vary
together in them. So the robustness analysis is not even a like-for-like replacement for
the question GATE 1 asked.

---

## 4. Comparisons, metrics and tests — frozen here, before results are seen

### 4.1 Distributions

`clean` = `test`; `easy`; `moderate` = `augmented_test_images_enhanced`; `hard`.
Label mapping stated in every output. Definitions and measured severity:
`docs/CORRUPTION_PROTOCOL.md`.

### 4.2 Metrics

Top-1 accuracy (primary), Top-5 accuracy, macro-F1, per-class precision/recall/F1,
confusion matrix. All computed from the frozen prediction-level records, never
recomputed by hand.

### 4.3 The comparisons that will be made

Fixed now, so the reported set cannot be chosen after seeing which ones look good.

| # | Comparison | Question |
|---|---|---|
| C1 | E5 vs A0 | Efficient AE-TFPE vs the fair baseline |
| C2 | E5 vs M1, M2, M3 | the GATE 1 question, re-measured on test |
| C3 | E5 vs E3 | AE contribution, Efficient side — **confounded**, also changes fusion space |
| C4 | E5 vs E7 | 28×28 vs 7×7 grid |
| C5 | **A5 vs D1** | **the denoising objective — the cleanest available test** |
| C6 | A5 vs A3 | AE contribution, Original side |
| C7 | A5 vs F1, F2, F4 | AE fusion vs addition / concatenation / attention — **F2 also modifies the classifier stem** |
| C8 | A5 vs E5 | Original vs Efficient formulation |
| C9 | A0 vs B2 | our baseline vs an external lightweight baseline |
| C10 | each arm, clean → easy → moderate → hard | robustness degradation profile |

Every comparison is reported whether favourable or not. No comparison is added after
results are seen without being labelled **post-hoc**.

### 4.4 Statistical treatment

Two sources of uncertainty, never conflated:

- **A. Training-seed variability — UNAVAILABLE.** One seed (0) per arm. No multi-seed
  claim, no significance statement about architectural superiority that would require
  retraining variance. Stated wherever results appear.
- **B. Paired test-sample uncertainty — MEASURED.** Both models see the same test
  samples, so **model-vs-model comparison within one distribution is paired** and is
  valid at one seed:
  - **McNemar's test** (exact binomial for small discordant counts) on paired top-1
    correctness;
  - **paired bootstrap** 95 % confidence intervals (10,000 resamples, seed 0) for
    differences in accuracy and macro-F1.

  These quantify **test-sample uncertainty only** and **do not substitute for multi-seed
  retraining**. That sentence accompanies every interval reported.

- **Cross-distribution pairing.** The clean↔augmented mapping is **PROVEN**
  (`docs/CORRUPTION_PROTOCOL.md` §6), so same-model clean-vs-corrupted paired tests are
  also admissible, subject to the PNG/JPEG and resize confounds recorded there.

- **Multiplicity.** C1–C10 span many pairwise tests. Holm–Bonferroni correction is
  applied within each distribution, and both raw and adjusted p-values are reported.

### 4.5 Decision rules fixed in advance

1. A difference is called **detectable** only if the paired bootstrap CI excludes zero
   **and** the Holm-adjusted McNemar p < 0.05.
2. A detectable difference is still reported as **single-seed**, never as established
   architectural superiority.
3. Where E5 loses, it is reported in the same words and the same prominence as where it wins.
4. Arms are never dropped, and classes are never excluded, to improve a number.
5. If the evidence contradicts a manuscript claim, **the manuscript changes.**

---

## 5. What this campaign can and cannot conclude

**Can:** clean and synthetic-augmentation test performance of 16 arms under one identical
protocol; measured parameter/FLOP/latency/memory cost; paired test-sample uncertainty for
model-vs-model comparisons; degradation profiles across three severity tiers.

**Cannot:** multi-seed robustness; isolated noise resilience (§3); a clean-accuracy
superiority claim for E5 (§1); per-corruption-family attribution; any claim about B1 or
B3, which were not trained; any 50-epoch claim about A2, which stopped at 26.

---

## 6. Addition — the Controlled Synthetic Corruption Benchmark

**Added 2 September 2026, after the corruption-construction audit, before any model's
test performance was inspected.** Like the rest of this document, it is **not a
pre-registration** — it is a specification frozen ahead of results.

### 6.1 Why it was added

`docs/CORRUPTION_PROTOCOL.md` established, by measurement, that the pre-existing
Easy / Moderate / Hard sets mix photometric corruption with geometric augmentation:
roughly **84 % of `hard` images are flipped or rotated**, and the tiers also carry
occlusion (CoarseDropout, fog, rain, shadow) and colour shifts, all varying together.

Degradation on those sets therefore **cannot be attributed specifically to noise
robustness**, which is precisely what the reviewers' challenge concerns. The existing
sets remain a valid robustness measurement of a *different* kind; they simply cannot
carry the denoising claim.

### 6.2 What it is

**Name: Controlled Synthetic Corruption Benchmark.** It must never be called a
*real-world robustness benchmark*.

Six deterministic, label-preserving, **non-geometric** families at three fixed
severities (mild / moderate / severe), applied to the authoritative 8,335-image clean
test split only:

| Family | Parameter | mild | moderate | severe |
|---|---|---|---|---|
| Gaussian noise | sigma (0–255) | 8 | 18 | 35 |
| Impulse noise (salt-and-pepper) | ratio | 0.02 | 0.05 | 0.10 |
| Gaussian blur | sigma (px @224) | 0.8 | 1.6 | 3.0 |
| Brightness | multiplicative factor | 0.75 | 0.55 | 0.40 |
| Contrast | factor about per-image mean | 0.70 | 0.45 | 0.25 |
| JPEG compression | quality | 40 | 20 | 10 |

Impulse noise is the second noise family because it *is* the manuscript's own legacy
Type 1 corruption, so it speaks directly to the reviewers' noise challenge rather than
being chosen for convenience.

**Excluded by design:** rotation, flip, crop, translation, geometric warping. These
change pose or framing rather than degrading the image, and are the confound the
benchmark exists to remove.

Parameters were fixed a priori from the physical scale of each operator, before any
model was run against them, and **must not be retuned after seeing results**. No family
was added to improve the method's apparent performance.

### 6.3 Reproducibility

Every corrupted pixel is a pure function of

```
derive_seed(relative_path, family, severity, base=global_seed)   # blake2b-64
```

so iteration order, worker count and batch size cannot change a single pixel. Applied
on the fly rather than written to disk (150,030 images would be ~12 GB for no gain),
with the full specification — family, severity, parameters, derived seed and the
`sha256_array` pixel hash of every one of the 150,030 corrupted samples — frozen to
`results/controlled_corruptions/`, alongside the spec hash and the code commit.
Determinism was verified by regeneration: 0 mismatches.

### 6.4 Analytical separation — the two benchmarks are never merged

| | Benchmark | Measures |
|---|---|---|
| **A** | Clean / Easy / Moderate / Hard | synthetic **augmentation** robustness (geometry + photometry + occlusion + noise, entangled) |
| **B** | Controlled Synthetic Corruption Benchmark | targeted **corruption / noise** robustness (non-geometric, per-family) |

They are reported in separate tables and are **never combined into a single
unexplained average**. B is an *additional* reviewer-driven analysis; it does **not**
replace A, and A is unchanged by its introduction.

Neither benchmark alters §1: **GATE 1 was not passed on clean validation.**

### 6.5 Models and comparisons

Primary: **A0, A5, D1, E5, B2**, plus **F2, F4** if runtime permits. Remaining ablations
only if runtime permits and they materially support a reviewer response.

The most important comparison is **A5 vs D1** — the cleanest available test of whether
the denoising objective improves robustness, since the two differ in the training
objective alone. Also reported: **A5 vs A0**, **E5 vs A0**, **E5 vs B2**.

### 6.6 Metrics and statistics

Per model × family × severity: **Top-1**, **Macro-F1**, **absolute degradation from
Clean**, **retention from Clean**. Every family is reported separately; a mean across
families appears only as a clearly labelled secondary column, because averaging
Gaussian noise with JPEG and brightness hides the failure modes the benchmark exists to
expose. Severity curves where useful.

Paired model-vs-model statistics (McNemar + paired bootstrap, Holm-corrected) are
computed **within each corruption distribution** — valid at a single seed because both
models see identical corrupted images. Training-seed variability remains **UNAVAILABLE**.
