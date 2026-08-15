# Provenance Matrix and Scientific Protocol Review

**Date:** 15 August 2026 · **Status:** pre-campaign review. Nothing committed, no training launched.

Every implementation decision is classified as:

- **RECOVERED** — supported by a surviving historical artifact.
- **RECONSTRUCTED** — the original could not be recovered; this is a documented rebuild of the intent.
- **NEW REVISION PROTOCOL** — introduced by this revision, with no historical counterpart, to satisfy a reviewer or fix a methodological defect.

---

## 1. Provenance matrix

### 1.1 Method components

| Decision | Status | Evidence | Confidence | Manuscript implication |
|---|---|---|---|---|
| ViT checkpoint `google/vit-base-patch16-224-in21k` | **RECOVERED** | `feature_extractor_folder.py:10` — exact string | HIGH | May be stated as the original choice |
| No transformer forward pass ever ran | **RECOVERED** | `ViTForImageClassification` imported `:3`, never called; output reproduced byte-exactly by the preprocessor alone | HIGH | §3.2 as written describes something that was never executed |
| Legacy transform = `0.2·LUT(x) + 0.8·x` | **RECOVERED** | MAE **0.0000**, exact byte match vs `features_extracted_dataset-(0.2-0.8)`; all other α gave MAE ≥ 7.17 | HIGH | The historical "feature fusion" is a zero-parameter pointwise LUT |
| Unclamped uint8 wrap in the round-trip | **RECOVERED** | torchvision `to_pil_image` does `(npimg*255).astype(np.uint8)`; reproduced exactly | HIGH | The mechanism is a non-monotonic contrast remap, not a feature |
| A second blend ratio (0.1/0.9) also existed | **RECOVERED** | `validation.py` → `yolo_labels_lant_leaf_disease_vit_0.1-0.9/data.yaml` | MEDIUM | Which ratio backs which table is unrecoverable — must be disclosed |
| Classifier = YOLOv8n-cls, pretrained | **RECOVERED** | `log-org-280223`; reproduced: 1,488,247 params, 156/158 transfer | HIGH | §4.2.2's "randomly initialized" claim is false |
| Optimizer AdamW, lr 7.14e-4, seed 0, 30 epochs, batch 128 | **RECOVERED** | `log-org-280223:2,26–27,184` | HIGH | §4.2.3's lr 0.01 / momentum 0.937 / 50 epochs are all contradicted |
| Split 38,584 / 8,340 / 8,335 | **RECOVERED** | log + dataset on disk match exactly | HIGH | The "61,486 images" claim is wrong; 55,259 is correct |
| 39 classes = PlantVillage 38 + `Background_without_leaves` | **RECOVERED** | derived from archived `image_id` prefixes | HIGH | Closes the §4.1 placeholder; the background class must be disclosed |
| PE-RGB construction | **RECONSTRUCTED** | nothing exists; machine-wide grep negative | HIGH *(that it is absent)* | §3.1/Eq. 1 must be labelled as newly implemented |
| Transformer usage (frozen ViT-B/16, patch tokens → 3-ch map) | **RECONSTRUCTED** | no forward pass ever existed | HIGH | §3.2 must report the new configuration, not an imagined one |
| Auto-encoder architecture | **RECONSTRUCTED** | no `nn.Module` anywhere on the machine | HIGH | §3.3 must describe the new AE |
| AE type = stacked sparse **denoising** | **RECONSTRUCTED** | chosen so terminology follows code | HIGH | The title/§1/§3.3 three-way naming conflict must be resolved to this one term |
| AE objective (MSE + KL + L2) | **RECONSTRUCTED** | Eqs. 5–6 are the intent; their notation is broken | HIGH | Equations must be re-typeset |
| YOLO interface = reconstructed RGB image | **RECONSTRUCTED**, consistent with RECOVERED behaviour | historical pipeline fed PNGs to a stock loader | HIGH | §3.4 must state the interface explicitly |
| `ae_loss_weight = 10.0` | **RECONSTRUCTED** | derived from loss-magnitude ratio; **not yet validation-justified** | LOW until Stage V1 | Must be reported as a hyperparameter with its selection procedure |
| `ae_warmup_epochs = 3` | **RECONSTRUCTED** | classical stacked-AE pretraining; **not yet validation-justified** | LOW until Stage V1 | Same |
| `pe_gamma = 0.1`, `pe_type = sincos1d` | **RECONSTRUCTED** | a priori; untuned | MEDIUM | Report as a fixed design choice, not a tuned one |
| AE latent `[128, 28, 28]` | **RECONSTRUCTED** | a priori | MEDIUM | Report as chosen, not searched |

### 1.2 Corruptions

| Decision | Status | Evidence | Confidence | Manuscript implication |
|---|---|---|---|---|
| Six pepper ratios 0.02–0.50 existed | **RECOVERED** | `evaluation_results/*/0.02-noise … 0.5-noise`; `calculate_top1.py` references `0.5-noise/labels/test` | HIGH | Severity *values* are historical |
| Combined noise+transparency sets existed | **RECOVERED** | `Plant_Handled-hard-{0.02..0.4}`, `Plant_Handled-medium` | MEDIUM | Only names survive; datasets are gone |
| Pepper definition (salt/pepper split, per-pixel vs per-channel) | **RECONSTRUCTED** | no code; §4.3.1 and §4.6 contradict each other | LOW | Must be labelled reconstructed; not comparable to Table 2 |
| Transparency α semantics | **RECONSTRUCTED** | §4.3.2 self-contradictory | LOW | Same |
| Type 3 order (noise → overlay) | **RECONSTRUCTED** | §4.3.3 states the order in prose | MEDIUM | Order is documented, parameters are not |
| Gaussian noise / blur / brightness / JPEG severities | **NEW REVISION PROTOCOL** | Reviewer #11 request | — | New table; no historical counterpart |
| Deterministic per-image seeding + sha256 manifest | **NEW REVISION PROTOCOL** | fixes the unseeded historical TTA defect | — | Enables the fairness claim |
| Corruptions on **test only**; train/val clean | **NEW REVISION PROTOCOL** | historical framing (§4.3) called them training augmentation | — | §4.3 must be reframed |
| Separate **val** corruption set for calibration | **NEW REVISION PROTOCOL** | added during this review (§3) | — | Guarantees the test set is unobserved during development |

### 1.3 Evaluation and training protocol

| Decision | Status | Evidence | Confidence | Manuscript implication |
|---|---|---|---|---|
| Historical eval used a **detection** validator | **RECOVERED** | `validation.ipynb` cell 0 `save_json=True, max_det=1, conf=0.25`; archived JSON carries `bbox`; the Ultralytics *classification* validator has no `conf`, no `max_det`, no `save_json` | HIGH | The reported "Top-1" is not classification top-1 — see `EVALUATION_PROTOCOLS.md` |
| Images with no detection counted as errors | **RECOVERED** | `calculate_top1.py` divides by the count of label files | HIGH | Confidence rejection is baked into the historical numbers |
| Historical numbers are **val**, not test | **RECOVERED** | run dirs named `_val`/`_val2`; `log-org-280223` `split=val` | HIGH | Any "test set" claim must be corrected |
| §4.6.2 YOLOv7 = 9.08% | **RECOVERED** | 757/8,340 = 9.077% reproduces exactly | HIGH | Traceable |
| §4.6.2 proposed+YOLOv7 = 14.8% | **NOT RECOVERED** | archived VIT runs give 16.40% / 16.55% | — | Reconcile or withdraw |
| Single frozen training protocol for all arms | **NEW REVISION PROTOCOL** | Reviewer #10.7 | — | Enables the fairness claim |
| Custom PyTorch loop wrapping `ClassificationModel` | **NEW REVISION PROTOCOL** | verified faithful: 1,488,247 params, 156/158 transfer | — | Must be disclosed as a deviation from the Ultralytics trainer |
| Checkpoint = best validation top-1 | **NEW REVISION PROTOCOL** | historical selection rule unknown | — | Report the rule |
| All arms output `[0,1]` to the classifier | **NEW REVISION PROTOCOL** | fixes a fairness bug found in shape validation | — | Report as a design constraint |

---

## 2. Test-set leakage audit

`scripts/train.py` contains **no reference to the test split** — verified by grep. Training and checkpoint selection use train/val only. The audit below covers every decision that could have been influenced by test data.

| Decision | Selected using | Leakage? | Action |
|---|---|---|---|
| Fusion output range (sigmoid fix) | shape report on random tensors, no data at all | **No** | None |
| AE architecture, latent size, `pe_gamma`, `pe_type` | a priori, before any run | **No** | None |
| Checkpoint selection | validation top-1 only; warm-up epochs excluded | **No** | None |
| `ae_loss_weight = 10.0` | ratio of training-loss magnitudes; confirmed on **validation** top-1 | **No test leakage**, but not yet a defensible selection | **Stage V1 below** |
| `ae_warmup_epochs = 3` | same | Same | **Stage V1 below** |
| Corruption severities | fixed a priori in `configs/corruptions.yaml` **before** any evaluation | **No** — values were never changed afterwards | Frozen; see `CORRUPTION_SPEC.md` |
| Corruption severity *sanity check* | **the smoke A0 model was evaluated on the corrupted TEST split** | **Soft exposure — declared** | See below |

### 2.1 The one exposure, stated plainly

During validation I ran `evaluate.py` on a 2-epoch smoke model and read its top-1 across all 26 corrupted **test** conditions. No parameter was changed as a result, and the model was discarded. But the honest position is that test-derived numbers were observed during development, and that must not become the norm.

**Corrective action, implemented in this review:** `generate_corruptions.py` now takes `--split`. A validation corruption set (`results/corruptions_val`) is generated alongside the frozen test benchmark, and **every** future calibration, sanity check or debugging pass uses the validation set. The test benchmark is touched exactly once per model, at Stage 8.

### 2.2 Stage V1 — the smallest experiment that freezes the two AE hyperparameters

Two values need validation-based justification, not a search. One factor varied at a time against the chosen setting:

| Run | `ae_loss_weight` | `ae_warmup_epochs` | Tests |
|---|---|---|---|
| V1a | 10 | 3 | the chosen setting |
| V1b | 1 | 3 | is the reweighting necessary? |
| V1c | 10 | 0 | is the warm-up necessary? |

- **Data:** 20% stratified subset of the *training* split; evaluated on the full **validation** split and on `corruptions_val` at `pepper/030`.
- **Budget:** 10 epochs each, ~20 min per run on a T4 → **~1 GPU-hour total**.
- **Selection rule, pre-committed:** pick the setting with the highest **validation** top-1. If two are within 0.5 pp, prefer the simpler one (lower weight, fewer warm-up epochs). Robustness on `corruptions_val` is reported but does **not** drive selection.
- **Outcome:** the winner is frozen into `_base.yaml` and never revisited. If V1b or V1c wins, the current defaults change accordingly — that is the point of running it.

This is three short runs, not a sweep, and it converts two LOW-confidence reconstructions into documented, validation-justified choices.

---

## 3. Design flaws found in this review, and their fixes

### 3.1 The fusion table was confounded by training-noise exposure — **fixed**

`A5_aetfpe_full` uses a denoising objective, so its AE sees synthetic noise during training. `F1_fusion_add`, `F2_fusion_concat` and `F4_fusion_attention` do not. Comparing A5 against them would have attributed to *"AE fusion"* a gain partly caused by noise exposure. That would have been a genuine confound in the headline fusion table Reviewer #12 asked for.

**Fix:** added `D1_ae_standard` — identical to A5 in every respect except `ae_denoising: false` (verified: the two configs differ in exactly one flag; both have 87,549,150 parameters). The fusion table now uses D1, whose training data is clean like every conventional operator's.

| Comparison | What it isolates | Valid? |
|---|---|---|
| D1 vs F1 / F2 / F3 / F4 | the auto-encoder **as a fusion mechanism**, at equal noise exposure | **Yes — use this for the fusion table** |
| A5 vs D1 | the **denoising objective** (Reviewer #10.4's specific claim) | Yes |
| A5 vs M3 | the AE given equal noise exposure | Yes |
| A5 vs A0 / A1 / A2 / A3 | *confounded* — differs in both AE and noise exposure | **Must be read alongside D1** |

### 3.2 PE-RGB is theoretically weak in image space — **flagged, not fixed**

This one can affect the manuscript's contribution claim, so it needs stating before the campaign.

The positional encoding is a **fixed additive field, identical for every image**. It carries no per-image information, and a convolutional classifier can learn to subtract a constant offset. On that reasoning `A1_pe_only` should be statistically indistinguishable from `A0_baseline_rgb`, and the PE branch should contribute little inside A5 either.

That is a legitimate experimental outcome and Reviewer #12.1 explicitly asked for it. But the revision must not claim PE "preserves spatial structure" if A1 ≈ A0. Two honest options, to be chosen **after** seeing A1 on validation:

1. Report the null result and reduce the PE claim to a component that was tested and found not to contribute.
2. Replace the additive field with a formulation that is at least input-dependent (e.g. concatenating PE as extra channels so the classifier cannot trivially cancel it). This changes the method and must be declared a revision-time change, not a recovery.

**Recommendation:** run A1 as specified, and pre-commit to option 1 unless A1 beats A0 on validation by a margin exceeding run-to-run noise. Do not silently redesign PE after seeing a null result.

### 3.3 Top-5 is undefined under the historical protocol — see `EVALUATION_PROTOCOLS.md`

The manuscript's Tables 2–3 report Top-5, but the recovered evaluation used `max_det=1`, which can only produce one prediction per image. Tables 2–3 and Fig. 9 therefore cannot both come from the same pipeline. This is a pre-existing inconsistency in the original work, not something the revision introduced, and it must be disclosed.

---

## 3a. FROZEN run matrix

Sixteen unique trainings, grouped by the question each group answers. Reuse is
explicit, so the count of *actual* trainings is unambiguous.

### Group 1 — Component ablation (6 unique)

| ID | PE | TF | AE | Fusion | Answers |
|---|---|---|---|---|---|
| A0_baseline_rgb | – | – | – | – | reference |
| A1_pe_only | ✓ | – | – | – | #12.1 |
| A2_tf_only | – | ✓ | – | linear | #12.2 |
| A3_pe_tf_no_ae | ✓ | ✓ | – | linear | #11, #12.4 |
| A4_rgb_ae | – | – | ✓ | – | #12.3 |
| A5_aetfpe_full | ✓ | ✓ | ✓ | linear | the proposed method |

### Group 2 — Fusion ablation (3 unique + 2 reused = 5 rows)

| Row | Source | Unique training? |
|---|---|---|
| F1_fusion_add | own run | **yes** |
| F2_fusion_concat | own run (**modifies the classifier stem**) | **yes** |
| F3_fusion_linear | **= A3_pe_tf_no_ae** | no — reused |
| F4_fusion_attention | own run | **yes** |
| F5_fusion_ae | **= D1_ae_standard** | no — reused from Group 3 |

The AE row of this table is **D1, not A5**. A5's denoising objective exposes it to
synthetic noise that F1/F2/F4 never see; using A5 here would confound the fusion
mechanism with noise exposure (§3.1).

### Group 3 — Denoising-objective ablation (1 unique)

| ID | Differs from A5 by | Answers |
|---|---|---|
| D1_ae_standard | `ae_denoising: false` only — verified: one config key, identical 87,549,150 parameters | #10.4. `A5 − D1` isolates the denoising objective |

### Group 4 — Mechanism controls (3 unique)

| ID | Tests | Answers |
|---|---|---|
| M1_legacy_lut | the historical zero-parameter transform | #10.1, #10.4 |
| M2_photometric | any monotonic contrast remap | #10.4 |
| M3_aug_control | corruption-augmented training alone | #10.4, #10.7 |

### Group 5 — External baselines (3 unique)

| ID | Answers |
|---|---|
| B1_resnet50, B2_efficientnet_b0, B3_vit_b16 | #10.7, #11 — one frozen protocol |

### Totals

| | Count |
|---|---|
| Config files | 16 |
| **Unique trainings** | **16** |
| Table rows served by reuse | 3 (F3 = A3, F5 = D1, and A5 appears in both Group 1 and Group 3) |
| Rows presented across all tables | 19 |

---

## 3b. FROZEN Stage V1

The only permitted candidates. No other combination may be run.

| Run | `ae_loss_weight` | `ae_warmup_epochs` |
|---|---|---|
| V1a | 10 | 3 |
| V1b | 1 | 3 |
| V1c | 10 | 0 |

| Setting | Value |
|---|---|
| Training data | 20% stratified subset (`--limit-train-per-class 200`) |
| Validation data | **100%** of the validation split (`--limit-val-per-class` unset) |
| Robustness data | `corruptions_val` only — never the test benchmark |
| Epochs | 10 |
| Selection | highest **validation top-1**; if two are within **0.5 pp**, choose the simpler configuration (lower weight, then fewer warm-up epochs) |
| Budget | ~1 GPU-hour total |

**No further tuning is permitted after V1** unless training is demonstrably
broken — meaning a run diverges, produces NaNs, or sits at chance. "The result
was disappointing" is not demonstrably broken.

---

## 4. Re-audit of the training-run matrix

16 unique runs: 6 ablation + 4 fusion + 3 mechanism + 3 baselines. Cost assumes a T4 at ~2.0–2.5 h per YOLO-arm run (30 epochs × 38,584 images at 224 px, extrapolated from the recovered 1.98 h baseline).

| Run | Scientific question | Reviewer | Genuinely unique? | Reuse | Cost (T4) |
|---|---|---|---|---|---|
| **A0** baseline | Reference point for every claim | #10.7, #11 | Yes | — | 2.0 h |
| **A1** +PE | Does positional encoding contribute? | #12.1 | Yes | — | 2.0 h |
| **A2** +TF | Does the transformer branch contribute? | #12.2 | Yes | — | 2.5 h |
| **A3** PE+TF, no AE | Do the two branches help without the AE? | #11, #12.4 | Yes | **= F3 (concat+linear)** | 2.5 h |
| **A4** RGB+AE | Does the AE help without the fusion? | #12.3 | Yes | — | 2.2 h |
| **A5** full AE-TFPE | The proposed method | #10, #11, #12 | Yes | — | 2.5 h |
| **D1** AE, clean→clean | AE as a fusion mechanism, unconfounded; and (with A5) the denoising objective | #10.3, #10.4, #12 | Yes — added by this review | = F5 for the fusion table | 2.5 h |
| **F1** addition | Conventional fusion baseline | #10.3, #12 | Yes | — | 2.5 h |
| **F2** concatenation | Conventional fusion baseline (**modifies the stem**) | #12 | Yes | — | 2.5 h |
| **F4** attention | Attention fusion baseline | #10.3, #12 | Yes | — | 2.5 h |
| **M1** legacy LUT | Reproduces the historical mechanism under the new protocol | #10.1, #10.4 | Yes | — | 2.0 h |
| **M2** photometric control | Is the wrap discontinuity doing the work, or any contrast remap? | #10.4 | Yes | — | 2.0 h |
| **M3** augmentation control | Is the gain just noise-augmented training? | #10.4, #10.7 | Yes | — | 2.0 h |
| **B1** ResNet-50 | Fair CNN baseline | #10.7, #11 | Yes | — | 3.5 h |
| **B2** EfficientNet-B0 | Fair CNN baseline | #10.7, #11 | Yes | — | 2.5 h |
| **B3** ViT-B/16 | Fair transformer baseline | #10.7, #11 | Yes | — | 7.0 h |

**Deduplicated away:** `F3_fusion_linear` (= A3), `F5_fusion_ae` (= A5), `F5_fusion_ae_clean` (= D1) — three fusion-table rows served by existing runs.

**Nothing removed.** Every run maps to an explicit reviewer request. The two weakest candidates were examined and kept:

- **A1** is expected to be a null result (§3.2), but Reviewer #12.1 asks for it by name, and a documented null is more valuable than a missing row.
- **F2** duplicates F3 conceptually, but Reviewer #12 lists plain concatenation and concatenation+projection as separate comparators, and F2 is the only arm that modifies the classifier — which is itself worth reporting.

**Total ≈ 40 T4-hours**, or ≈ 13–16 h on an A100/L4, plus ~1 h for Stage V1 and ~4 h for evaluation.

---

## 5. Why M1 / M2 / M3 exist, and the protocol trap they must avoid

The archived YOLOv7 pair gives 9.08% → 16.40% under the historical protocol: a **1.81× robustness gain produced by a zero-parameter pointwise lookup table**. The abstract claims "approximately 1.9×" for the full AE-TFPE. If a fixed LUT already delivers essentially the paper's headline number, then the auto-encoder's contribution is unestablished.

The three controls decompose the competing explanations:

| Arm | Hypothesis it tests | If it matches A5 |
|---|---|---|
| **M1** legacy LUT `0.2·LUT(x)+0.8·x` | The gain comes from the historical transform itself | The contribution is a lookup table, not a learned representation |
| **M2** monotonic gamma, matched deviation, no wrap | The gain comes from *any* pointwise contrast remap | The wrap discontinuities are irrelevant; the mechanism is contrast |
| **M3** clean RGB + corruption-augmented training | The gain comes from noise exposure during training | The contribution is data augmentation, not architecture |

A5 must beat all three for the paper's claim to stand.

### 5.1 The protocol trap

**M1 reproduces the historical transform, not the historical evaluation.** It is trained and scored under the new revised classification protocol. Its number is therefore **not comparable to the historical 16.40%**, which came from a detection validator with `conf=0.25` rejection on a different (now lost) corrupted dataset.

Three rules follow, and they must hold in every table:

1. Never place M1's new number in the same table as the historical 16.40% without an explicit protocol column.
2. All of A0, A5, M1, M2, M3 are trained and evaluated under one protocol on one frozen benchmark, so comparisons *among them* are valid.
3. The historical 1.81× is cited as **motivation** for running M1 — it is not evidence about M1's outcome.

### 5.2 Fairness inside the mechanism gate

M3 receives training-time corruption; A0, M1 and M2 do not; A5 does (via its denoising objective). So:

- **A5 vs M3** is the clean AE-vs-augmentation test — both see noise.
- **A5 vs M1/M2** is confounded by noise exposure and must be read together with **D1 vs M1/M2**, where neither side sees noise.

Both comparisons are reported. Neither is presented alone.
