# Implementation Validation

**Date:** 15 August 2026 · **Environment:** local MacBook (Apple Silicon, MPS), development only
**Status:** first implementation deliverable complete. **The full ablation campaign has NOT been launched.**

Everything below was produced on the development laptop and is a *correctness*
check, not a result. No number in this document may be used as evidence in the
manuscript. Training, evaluation and all runtime measurements happen on Colab
with CUDA — see [`RUNNING.md`](RUNNING.md).

---

## 1. Architecture

Pipeline, with measured shapes (`scripts/check_shapes.py` →
`results/validation/shape_report.json`):

```
RGB          [B, 3, 224, 224]  float in [0,1]
 -> PE-RGB   [B, 3, 224, 224]  x + 0.1 * PE(16x16 grid), clamped
 -> ViT-B/16 [B, 197, 768] -> drop CLS -> [B, 196, 768] -> [B, 768, 14, 14]
 -> TF-RGB   [B, 3, 224, 224]  1x1 conv + BN + sigmoid, bilinear upsample
 -> fusion   [B, 3, 224, 224]  (add / linear / attention)
             [B, 6, 224, 224]  (concat only)
 -> AE       [B, 128, 28, 28] latent -> [B, 3, 224, 224] reconstruction
 -> YOLOv8n-cls -> [B, 39]
```

The auto-encoder is a **stacked sparse denoising auto-encoder**: corrupted input,
clean reconstruction target, KL sparsity on latent channel means, L2 via weight
decay, three encoder and three decoder stages. Terminology follows the code, as
required.

### 1.1 Per-arm shapes and parameters — all 15 arms pass

| Arm | pre-AE | latent | classifier input | params total | trainable | front-end | output range |
|---|---|---|---|---|---|---|---|
| A0_baseline_rgb | [2,3,224,224] | – | [2,3,224,224] | 1,488,247 | 1,488,247 | 0 | [0.00, 1.00] |
| A1_pe_only | [2,3,224,224] | – | [2,3,224,224] | 1,488,247 | 1,488,247 | 0 | [0.00, 1.00] |
| A2_tf_only | [2,6,224,224] | – | [2,3,224,224] | 87,289,243 | 1,490,587 | 85,800,996 | [0.35, 0.62] |
| A3_pe_tf_no_ae | [2,6,224,224] | – | [2,3,224,224] | 87,289,243 | 1,490,587 | 85,800,996 | [0.31, 0.54] |
| A4_rgb_ae | [2,3,224,224] | [2,128,28,28] | [2,3,224,224] | 1,747,290 | 1,747,290 | 259,043 | [0.45, 0.55] |
| **A5_aetfpe_full** | [2,6,224,224] | [2,128,28,28] | [2,3,224,224] | 87,549,150 | 1,750,494 | 86,060,903 | [0.45, 0.55] |
| F1_fusion_add | [2,6,224,224] | – | [2,3,224,224] | 87,289,216 | 1,490,560 | 85,800,969 | [0.22, 0.80] |
| F2_fusion_concat | [2,6,224,224] | – | **[2,6,224,224]** | 87,289,648 | 1,490,992 | 85,800,969 | [0.00, 1.00] |
| F4_fusion_attention | [2,6,224,224] | – | [2,3,224,224] | 87,289,288 | 1,490,632 | 85,801,041 | [0.41, 0.62] |
| M1_legacy_lut | [2,3,224,224] | – | [2,3,224,224] | 1,488,247 | 1,488,247 | 0 | [0.01, 0.97] |
| M2_photometric | [2,3,224,224] | – | [2,3,224,224] | 1,488,247 | 1,488,247 | 0 | [0.00, 1.00] |
| M3_aug_control | [2,3,224,224] | – | [2,3,224,224] | 1,488,247 | 1,488,247 | 0 | [0.00, 1.00] |
| B1_resnet50 | – | – | [2,3,224,224] | 23,587,943 | 23,587,943 | 0 | [0.00, 1.00] |
| B2_efficientnet_b0 | – | – | [2,3,224,224] | 4,057,507 | 4,057,507 | 0 | [0.00, 1.00] |
| B3_vit_b16 | – | – | [2,3,224,224] | 85,828,647 | 85,828,647 | 0 | [0.00, 1.00] |

The ViT contributes 85.8 M **frozen** parameters; trainable counts stay near the
1.49 M baseline for every YOLO arm. The auto-encoder adds 259,043 parameters
(3-channel input) or 259,907 (6-channel).

### 1.2 Fidelity checks against the historical log

| Quantity | `log-org-280223` | This implementation |
|---|---|---|
| YOLOv8n-cls parameters | 1,488,247 | **1,488,247** |
| Pretrained weight transfer | "Transferred 156/158 items" | **156/158** |
| Classes | 39 | 39 |
| Split sizes | 38,584 / 8,340 / 8,335 | 38,584 / 8,340 / 8,335 |

Both parameter count and transfer ratio reproduce exactly, which is the evidence
that the custom training loop wraps the same classifier the original used.

### 1.3 Bug found and fixed during validation

The first shape check showed `A3_pe_tf_no_ae` emitting `[-0.2504, 0.0777]` while
the baseline emitted `[0.0, 1.0]`. `LinearProjectionFusion` and `AttentionFusion`
ended in SiLU, so those arms handed the pretrained stem a different input
distribution from the baseline — a fairness bug that would have confounded the
whole ablation with a normalisation artefact. Both now end in a sigmoid, and
**every arm is verified in `[0, 1]`**. Caught before any training was launched.

---

## 2. Dataset

`DATA_ROOT` → `Plant_leaf_diseases_dataset`

| Split | Images | Classes |
|---|---|---|
| train | 38,584 | 39 |
| val | 8,340 | 39 |
| test | 8,335 | 39 |

39 classes = PlantVillage's 38 disease classes + `Background_without_leaves`.
Counts match `log-org-280223` exactly.

### 2.1 There are two dataset copies with different splits — use the right one

A second copy sits inside the repository at `dataset/Plant_leaf_diseases_dataset`
(3.7 GB, now git-ignored). Its `data.yaml` points at
`FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8/`, i.e. it belongs to the
**sibling attention-mechanism paper**, and its split differs:

| Copy | train | val | test | Extra |
|---|---|---|---|---|
| `~/…/Vision Transformer/dataset/…` (**use this**) | 38,584 | **8,340** | **8,335** | — |
| `<repo>/dataset/…` (sibling paper) | 38,584 | **8,346** | **8,334** | `augmented_test_images_{easy,enhanced,hardest}`, 8,335 each |

Only the first matches `log-org-280223`. `configs/local.yaml` points at it.
Pointing `DATA_ROOT` at the in-repo copy would silently shift 6 images between
val and test and break comparability with every historical number.

The three `augmented_test_images_*` directories are the sibling paper's
pre-generated TTA sets — the same ones behind
`evaluation-results/{easy,medium,hardest}_augment_evaluation_report.csv`. They
were produced by the unseeded Albumentations protocol and **must not** be used as
this paper's robustness benchmark.

Corruptions are applied **only** to the test split. Train and val stay clean,
except in arm M3, which applies its augmentation in the dataloader and never
touches the frozen files.

---

## 3. Corruption pipeline

### 3.1 Determinism — verified

```
python scripts/generate_corruptions.py --limit-per-class 2
python scripts/generate_corruptions.py --limit-per-class 2 --verify
  -> verified 2028 files, 0 mismatches
```

2,028 files (26 configurations × 78 images) were regenerated in memory and
compared against their recorded sha256. **Zero mismatches.**

The RNG for each image is derived from `(seed, relative_path, corruption,
severity)` via blake2b, not from iteration order, worker count, or Python's
salted `hash()` — so a corrupted file is reproducible in isolation and identical
across macOS and Linux.

### 3.2 Manifest

`results/corruptions/corruption_manifest.csv`, sha256
`decc3a254f486d31f3520cb7ac3ffb05e484bf99d7db5e714e6609090daad7aa`
(2,028 rows for the smoke sample). Columns: `original_path, corrupted_path,
class, corruption, severity, seed, parameters, checksum`. Example row:

```
corrupted_path : pepper/002/Corn___healthy/image (100).png
corruption     : pepper        severity : 002        seed : 0
parameters     : {"per_channel": false, "ratio": 0.02, "salt_vs_pepper": 0.5}
checksum       : 296c345ca1b2e2e5bd18b1a3d00bfec9...
```

`evaluate.py` records the manifest's own sha256 in every result, so two runs can
be *proven* to have been scored on identical bytes.

### 3.3 Configurations

**26 configurations**, in both the smoke sample and the full plan.

| Family | Severities |
|---|---|
| clean | 1 |
| pepper (Type 1) | 0.02, 0.10, 0.20, 0.30, 0.40, 0.50 |
| transparency 70% (Type 2) | 1 |
| pepper + transparency (Type 3) | 0.02 … 0.50 |
| gaussian_noise | σ = 10, 25, 50 |
| gaussian_blur | radius = 1, 2, 4 |
| brightness | ×0.7, ×0.5, ×0.3 |
| jpeg | quality 40, 20, 10 |

Severity calibration was checked on the smoke baseline and behaves sensibly —
monotone degradation within every family, pepper hardest, transparency mildest.
Parameters were fixed **before** any model was evaluated and must not be retuned.

**Disk:** the smoke sample (2 images/class, 2,028 files) is 196 MB, i.e.
94.5 KB/file. The full benchmark is 26 × 8,335 = **216,710 files ≈ 21 GB** —
generate it on Colab, not on the laptop (12 GB free at time of writing).

---

## 4. Smoke tests

All intentionally tiny: 8 images per class, batch 32, MPS.

### 4.1 Baseline trains

```
python scripts/train.py --config configs/baseline_rgb.yaml \
    --epochs 2 --limit-per-class 8 --batch-size 32
```

| Epoch | train loss | train top-1 | val top-1 |
|---|---|---|---|
| 1 | 3.6672 | 0.0321 | 0.0962 |
| 2 | 3.1893 | 0.2981 | **0.4359** |

### 4.2 Full AE-TFPE trains — after a deadlock was found and fixed

The first attempt sat at chance (0.0256 = 1/39) for both epochs with a flat
reconstruction loss. Diagnosis: reconstruction MSE (~0.04) and cross-entropy
(~3.8) differ by two orders of magnitude, so the AE received almost no gradient
pressure while the classifier could not learn from an unreconstructed input.

Two fixes, both classical and both fixed a priori rather than tuned on results:
`ae_loss_weight = 10.0` (makes the terms comparable at the start of joint
training) and `ae_warmup_epochs = 3` (reconstruction-only epochs — exactly the
pretraining a *stacked* auto-encoder classically receives). Checkpoints are never
selected from warm-up epochs.

```
python scripts/train.py --config configs/aetfpe_full.yaml \
    --epochs 8 --limit-per-class 8 --batch-size 32
```

| Epoch | Stage | train loss | val top-1 |
|---|---|---|---|
| 1–3 | ae_warmup | 1.5565 → 1.0083 | – |
| 4 | joint | 4.7407 | 0.0609 |
| 5 | joint | 4.6081 | 0.0769 |
| 6 | joint | 4.3618 | 0.1218 |
| 7 | joint | 4.1214 | 0.1795 |
| 8 | joint | 4.1017 | **0.2372** |

Reconstruction loss falls during warm-up, then classification climbs steadily
once joint training begins. The deadlock is resolved.

### 4.3 Full evaluation pass — 27 conditions

```
python scripts/evaluate.py --run results/validation/A0_smoke --limit-per-class 4
```

| Condition | top-1 | top-5 |
|---|---|---|
| clean test split | 0.4167 | 0.7500 |
| clean/none | 0.3974 | 0.7564 |
| transparency/70 | 0.3333 | 0.6410 |
| brightness easy / medium / hard | 0.3718 / 0.3846 / 0.2692 | 0.7692 / 0.7436 / 0.6154 |
| jpeg easy / medium / hard | 0.3590 / 0.2692 / 0.1923 | 0.6667 / 0.5513 / 0.5128 |
| gaussian_blur easy / medium / hard | 0.3974 / 0.1667 / 0.0769 | 0.6282 / 0.4359 / 0.2308 |
| gaussian_noise easy / medium / hard | 0.2692 / 0.1154 / 0.0769 | 0.6154 / 0.4744 / 0.2436 |
| pepper 0.02 → 0.50 | 0.2436 → 0.0256 | 0.5641 → 0.1154 |
| pepper+transparency 0.02 → 0.50 | 0.2308 → 0.0256 | 0.5000 → 0.1410 |

Every condition read the frozen files; the manifest hash is recorded in
`eval_summary.json`.

### 4.4 Analyses run end to end

| Script | Result |
|---|---|
| `analyze_complexity.py --skip-latency` | 15 arms, params + FLOPs written |
| `confusion_matrix.py` | matrix CSV, per-class CSV and PNG written |
| `analyze_latent_stability.py` | drift statistics + embeddings written |
| `plot_latents.py` | t-SNE figure written |

**Latent stability — pipeline check only.** On the 8-epoch smoke model, the AE
latent's relative L2 drift under `pepper/030` was 0.171 against 0.561 before the
AE (ratio **0.306**), cosine similarity 0.987 vs 0.856. This is exactly the
measurement Reviewer #10 asked for, and the pipeline produces it correctly — but
**the model is undertrained on 8 images per class and both silhouette scores are
negative, so these values carry no scientific weight.** They will be recomputed
on Colab from a fully trained model.

### 4.5 Complexity — hardware-independent fields only

| Arm | Params | Trainable | GFLOPs @224 |
|---|---|---|---|
| A0 / A1 / M1 / M2 / M3 | 1,488,247 | 1,488,247 | 0.412 |
| A4_rgb_ae | 1,747,290 | 1,747,290 | 2.473 |
| A2 / A3 | 87,289,243 | 1,490,587 | 34.141 |
| F1 / F2 / F4 | ~87,289,300 | ~1,490,700 | ~34.14 |
| A5_aetfpe_full | 87,549,150 | 1,750,494 | 36.225 |
| B1_resnet50 | 23,587,943 | 23,587,943 | 8.264 |
| B2_efficientnet_b0 | 4,057,507 | 4,057,507 | 0.828 |
| B3_vit_b16 | 85,828,647 | 85,828,647 | 22.571 |

**Resolution matters.** 0.412 GFLOPs is YOLOv8n-cls at 224×224 counting 2×MACs.
Ultralytics' own `model.info()` prints "3.4 GFLOPs" because it measures at
640×640: 640²/224² = 8.16 and 0.412 × 8.16 = 3.36. The two agree once the
resolution is stated, and `img_size` is now a column in `complexity.csv`.

Latency, throughput and FPS were **not** measured. `analyze_complexity.py` stamps
every non-CUDA run `timings_reportable: false` with an explicit warning, and
`--skip-latency` omits them entirely. They must be collected in a single Colab
session on one GPU.

---

## 5. Environment portability — verified

The same config resolves correctly in both environments:

| | `data.root` | `corruption_root` |
|---|---|---|
| LOCAL (`configs/local.yaml`) | `/…/dataset/Plant_leaf_diseases_dataset` | `results/corruptions` |
| COLAB (`DATA_ROOT=/content/data/…`) | `/content/data/Plant_leaf_diseases_dataset` | `/content/output/corruptions` |

- No absolute user paths in any committed file — verified by grep over `src/`,
  `scripts/`, `configs/`.
- `configs/local.yaml`, `results/`, `data/`, `*.pt` are git-ignored.
- Device selection is CUDA → MPS → CPU, with no CUDA-only code paths.
- `scripts/colab_setup.sh` is POSIX bash, no Homebrew or macOS assumptions.
- `requirements.txt` now includes `ultralytics` (it was missing) and pins
  `numpy<2.0`, because numpy 2.x breaks the torch↔numpy bridge on torch 2.2
  builds — a failure hit and fixed during setup.

---

## 6. Deduplicated run matrix

`python scripts/print_run_matrix.py`

**9 unique training runs** cover the component ablation and fusion comparison
(within the 8–10 target): A0, A1, A2, A3, A4, A5, F1, F2, F4. `F3_fusion_linear`
is A3 and `F5_fusion_ae` is A5 — reused, not retrained. Plus 3 mechanism
controls and 3 fair baselines = **15 total**.

Recommended order (see `ARCHITECTURE_RECOVERY.md` §4.1): run **M1, M2, M3, A0,
A5 first**. The archived YOLOv7 evidence shows a 1.81× robustness gain already
produced by a zero-parameter lookup table, so the decisive question is whether
the auto-encoder beats it. If A5 does not clearly beat M1/M2/M3, the remaining
ten runs are not worth their compute.

---

## 7. Remaining issues

| # | Issue | Impact |
|---|---|---|
| 1 | Local disk has ~12 GB free; the full corruption benchmark needs **~21 GB** | Generate it on Colab. Archiving 21 GB to Drive may exceed quota — see COLAB_CAMPAIGN_PLAN.md Stage 2 for the checksum-only fallback |
| 2 | No full-dataset training has been run in this framework | Baseline must reproduce ≈0.99 top-1 on Colab before any ablation result is trusted. **This is the first gate.** |
| 3 | Latency/throughput unmeasured | Must be collected in one Colab session on one GPU |
| 4 | `ae_warmup_epochs=3` and `ae_loss_weight=10.0` validated only at smoke scale | Confirm the AE still converges at full scale; do not tune them on test results |
| 5 | Legacy Types 1–3 are reconstructions | Any comparison against the original Tables 2–3 must say so explicitly |
| 6 | Manuscript §4.6.2's 14.8 % figure does not reproduce | Reconcile or withdraw |
| 7 | Blueberry set is 74 images across 8 usable classes | Insufficient. Withdraw §4.6.4 and Figs. 11–12; state the limitation |
| 8 | Historical checkpoints exist at `weights/{ViT,YOLOv8-Base,Our-Proposed-Method}` | Not yet inspected. They may let some historical numbers be re-derived, but they were trained under the old, undocumented protocol and must not be mixed with new results |

---

## 8. Gate

**Do not launch the full ablation campaign until, on Colab:**

1. `scripts/check_shapes.py` passes on CUDA.
2. The full corruption benchmark generates and `--verify` reports 0 mismatches.
3. `A0_baseline_rgb` trains for the full 30 epochs and reaches roughly the
   historical 0.99 top-1. If it does not, the protocol reconstruction is wrong
   and every downstream number would be untrustworthy.

Only after those three pass should M1/M2/M3/A5 run, and only after the mechanism
gate should the remaining arms be scheduled.
