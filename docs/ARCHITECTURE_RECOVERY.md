# AE-TFPE — Architecture Recovery and Frozen Specification

**Status:** Phase 1 (evidence recovery) and Phase 2 (method freeze) complete.
**Date:** 15 August 2026 · **Deadline:** 5 September 2026
**Manuscript:** *Robust Feature Fusion Model for Plant Leaf Disease Classification* (MTAP, major revision)

This document has one job: state exactly what the historical implementation did,
exactly what the revised implementation does, and which of the two any given
claim rests on. Where the original could not be recovered, the reconstruction is
labelled **RECONSTRUCTED** and its justification is given. Nothing here is
inferred from the manuscript alone.

---

## 1. Evidence recovery

### 1.1 Artifacts examined

| Location | What it is |
|---|---|
| `feature_extractor_folder.py`, `feature_extractor.py`, `feature_fusion.py` | The only method code that ever existed in the repository |
| `log-org-280223` | One Ultralytics YOLOv8n-cls training log, Tesla T4, 30 epochs |
| `YOLO_V8_Classification.ipynb` | Repurposed scratch notebook — its executed cells train YOLO11n-cls on **UCF-Crime anomaly detection** (14 classes, 1,013,070 images, `imgsz=64`) |
| `evaluation-results/*.csv` | Belongs to a **different paper** (`FPT/YOLOv8-ResCBAM/`, attention-mechanism variants ECA/ResCBAM/GAM) |
| `evaluation-results/Augment Information.docx` | Albumentations TTA spec from that same other paper |
| `evaluation-results/archived/` | Four YOLOv7-tiny detection runs on the leaf data, `original` vs `VIT` |
| `~/Desktop/AI/Computer Vision/Vision Transformer/` | **The project's real working directory**, recovered during this phase |
| └ `dataset/Plant_leaf_diseases_dataset/` | 38,584 / 8,340 / 8,335 across 39 classes — **exactly the counts in `log-org-280223`** |
| └ `dataset/features_extracted_dataset-(0.2-0.8)/` | Partial (test split only, 2,165 images) — the surviving processed dataset |
| └ `dataset/features_extracted_dataset-org/` | Full, 4.6 GB — despite the name, **pixel-identical to the plain resized original** |
| └ `weights/{ViT,YOLOv8-Base,Our-Proposed-Method}/` | `best.pt` and `last.pt` for three historical models |
| └ `evaluation_results/` | 52 detection-metric CSVs across YOLOv6/9/10 × Original/VIT × noise ratio |
| └ `validation.py`, `validation.ipynb`, `calculate_top1.py`, `train.py` | The historical evaluation pipeline |
| └ `classified_dataset/` | 74 blueberry images, Roboflow-exported, 6 of 15 class folders empty |

### 1.2 Recovery results

| # | Target | Result | Confidence | Evidence |
|---|---|---|---|---|
| 1 | PE-RGB generation | **NOT FOUND** | HIGH (that it does not exist) | Tree-wide grep for `positional_encoding\|PositionalEncoding\|pos_encod` over every `.py`/`.ipynb` on the machine returns only vendored Ultralytics SAM source |
| 2 | Transformer architecture | **Only the checkpoint name** | HIGH | `ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")`, `feature_extractor_folder.py:10`. `ViTForImageClassification` imported at `:3`, **never called**. No forward pass exists |
| 3 | TF-RGB extraction | **RECOVERED — it is not a transformer feature** | HIGH, verified | `inputs.pixel_values`, `[1,3,224,224]`, `:13–16`. Round-tripped through `ToPILImage∘ToTensor∘Normalize∘ToPILImage` (`:44–58`), whose unclamped `(npimg*255).astype(np.uint8)` wraps modulo 256 twice |
| 4 | Feature fusion | **RECOVERED, byte-exact** | HIGH, verified | `(alpha*vit + beta*orig).astype("uint8")`, `:39`, called `alpha=0.2, beta=0.8` at `:136–138` |
| 5 | Autoencoder architecture | **DOES NOT EXIST** | HIGH | Grep for `autoencoder\|auto_encoder\|AutoEncoder` across the machine: two vendored SAM files, nothing else |
| 6 | Autoencoder objective | **DOES NOT EXIST** | HIGH | ditto |
| 7 | AE → YOLOv8 interface | **RECOVERED (no AE involved)** | HIGH | `img_combined.save(output_path)` writes `<name>_features.png` into an ImageFolder tree, `:139–146`; stock Ultralytics consumes it. No YOLO layer is modified anywhere |
| 8 | Corruption generation | **NOT FOUND** | HIGH (that no code survives) | Only directory *names* survive (`0.02-noise` … `0.5-noise`, `Plant_Handled-hard-*`). The datasets themselves are gone. The one surviving augmentation spec belongs to a different paper and contains neither salt-and-pepper nor transparency |
| 9 | Provenance of Tables 2–3 | **Protocol recovered; numbers partly traced** | MEDIUM–HIGH | See §1.4 |

### 1.3 Pixel-exact verification of the legacy transform

The reconstruction in `src/aetfpe/features/legacy_lut.py` was checked against the
surviving processed dataset:

| Target directory | Best (α, β) | Mean abs. error | Exact byte matches |
|---|---|---|---|
| `features_extracted_dataset-(0.2-0.8)/` | **α=0.2, β=0.8** | **0.0000** | 2 / 2 |
| `features_extracted_dataset-org/` | α=0.0, β=1.0 | 0.0000 | 24 / 24 |

Every other (α, β) tested gave MAE ≥ 7.17. The second row means
`features_extracted_dataset-org/` is the plain resized original despite its name,
so it is the **baseline** dataset in PNG form, not a processed one.

This upgrades the central forensic claim from inference to measurement: the
entire historical "feature extraction" is a fixed 256-entry pointwise lookup
table blended with the original image —

```
F(x)[c,i,j] = 0.2 · LUT[ x[c,i,j] ] + 0.8 · x[c,i,j]
```

— with **zero trainable parameters**, no receptive field, and therefore no
capacity to encode position or long-range context. The LUT is non-monotonic
(10 descending discontinuities) and only 159 of 256 output codes are reachable.

### 1.4 Provenance of the reported numbers

The historical evaluation protocol **was** recovered, with high confidence:

```python
# validation.ipynb, cell 0
results = model.val(data=data_yaml, split="test", save_json=True, plots=True,
                    max_det=1, conf=0.25, imgsz=224, device="mps")
```

```python
# calculate_top1.py — how "Top-1 Acc" in the manuscript was actually computed
label_folder = "0.5-noise/labels/test"
predictions  = json.load(open(".../best_predictions.json"))
# for each label file: compare its class_id against the single predicted category_id
# images with no detection are absent from the JSON and count as errors
top_1_accuracy = correct_predictions / total_labels
```

Consequences:

- Classification was performed **as object detection** with whole-image boxes,
  `max_det=1`, `conf=0.25`. Images below the confidence threshold produce no
  detection and are scored as errors. The manuscript reports these as
  classification accuracies without saying so.
- Recomputing this from the archived dumps gives **757 / 8,340 = 9.077 %** for
  `yolov7_tiny_original_224_val2`. Manuscript §4.6.2 reports YOLOv7 at
  **9.08 %** — an exact match to three significant figures.
- Its `VIT` counterpart gives **16.40 %**, but the manuscript reports **14.8 %**
  for "the proposed method with YOLOv7". **The baseline traces; the paired
  proposed figure does not.** This must be reconciled or withdrawn.
- The 52 surviving `evaluation_results/*.csv` files contain detection metrics
  (precision / recall / mAP50), and **none of them matches the top-1 values in
  Fig. 9**. They are a separate measurement campaign.
- `validation.py` points at `yolo_labels_lant_leaf_disease_vit_0.1-0.9/`, i.e. a
  **0.1 / 0.9** blend, whereas the surviving dataset is 0.2 / 0.8. At least two
  blend ratios were in use; which one backs which table is not recoverable.

### 1.5 Additional recovered facts

- **39 classes** = PlantVillage's 38 + `Background_without_leaves`. This closes
  the unfinished placeholder in §4.1. The presence of a non-leaf background class
  should be disclosed, since it inflates accuracy relative to a pure disease task.
- **All evaluation was on `val` or `test` directories named `_val`/`_val2`**;
  `log-org-280223` likewise ran `split=val`. Any number presented as test-set
  performance needs checking.
- The baseline log's resolved optimiser is **AdamW, lr 7.14e-4, momentum 0.9**,
  seed 0, deterministic, batch 128, 224 px, 30 epochs, `pretrained=True`
  ("Transferred 156/158 items"). The manuscript's §4.2.2–4.2.3 claims
  (random init, 50 epochs, lr 0.01, momentum 0.937) are contradicted by the log,
  which explicitly prints `ignoring 'lr0=0.01' and 'momentum=0.937'`.
- **Blueberry data partially exists**: 74 images across 9 non-empty class folders
  in `classified_dataset/`, Roboflow-exported with `_train`/`_test` suffixes.
  This is far too small and too incompletely documented to support §4.6.4. See §7.

---

## 2. Frozen specification of the revised method

### 2.1 End-to-end pipeline

Shapes below are measured, not asserted — `scripts/check_shapes.py` prints them
for all 15 arms and writes `results/validation/shape_report.json`.

```
Input RGB                     [B, 3, 224, 224]   float in [0,1]
  -> preprocessing            resize 224, ToTensor. NO normalisation here:
                              each component applies its own, so every arm
                              sees byte-identical pixels
  -> PE-RGB                   [B, 3, 224, 224]   x + gamma * PE, clamped to [0,1]
  -> transformer (ViT-B/16)   [B, 197, 768]      last_hidden_state
       drop CLS               [B, 196, 768]
       reshape                [B, 768, 14, 14]
       1x1 conv + BN + sigmoid[B, 3, 14, 14]
  -> TF-RGB                   [B, 3, 224, 224]   bilinear upsample, in [0,1]
  -> fusion                   [B, 3, 224, 224]   (add / linear / attention)
                              [B, 6, 224, 224]   (concat — the one arm that
                                                  modifies the classifier)
  -> autoencoder
       encoder                [B, 128, 28, 28]   latent, sigmoid
       decoder                [B, 3, 224, 224]   reconstruction, sigmoid
  -> YOLOv8-cls interface     [B, 3, 224, 224]   an ordinary image
  -> classifier               [B, 39]            logits
```

**Every arm hands the classifier a tensor in `[0, 1]`.** This is enforced, not
incidental: an unbounded fusion output would give that arm different input
statistics from the plain-RGB baseline, confounding the ablation with a
normalisation artefact. The first shape check caught exactly this bug
(`LinearProjectionFusion` was emitting `[-0.25, 0.08]`) and it was fixed before
any training was launched.

### 2.2 PE-RGB — **RECONSTRUCTED**

`src/aetfpe/features/positional_encoding.py`

| Property | Value |
|---|---|
| Input | `[B, 3, 224, 224]`, float in `[0, 1]` |
| Spatial partitioning | 16×16 patches → 14×14 = 196 patch grid |
| Formulation | `PE[p, 2j] = sin(p / n^(2j/d))`, `PE[p, 2j+1] = cos(p / n^(2j/d))`, `n = 10000`, `d = 3` |
| Broadcast | per-patch vector repeated over that patch's 16×16 pixels |
| Combination | `PE-RGB = clamp(x + gamma * PE, 0, 1)`, `gamma = 0.1` |
| Output | `[B, 3, 224, 224]`, float in `[0, 1]` |
| Trainable parameters | **0** (registered buffer) |

Manuscript Eq. (1) indexes a raster patch position `p` and an embedding
dimension `d`. Applied to an image rather than to token embeddings, `d = 3`.
That is the literal reading and it is the default (`pe_type: sincos1d`). A
`sincos2d` variant that splits the channel budget between row and column index
is also implemented, because a raster index discards 2-D structure; the choice is
a config field so the ablation can report it rather than bury it. `gamma = 0.1`
was chosen a priori as a small perturbation relative to the `[0,1]` signal range
and was **not** tuned on any result.

### 2.3 Transformer — **checkpoint RECOVERED, use RECONSTRUCTED**

`src/aetfpe/features/transformer_features.py`

| Property | Value | Source |
|---|---|---|
| Type | Vision Transformer, ViT-B/16 | RECONSTRUCTED |
| Library | `transformers.ViTModel` | RECONSTRUCTED |
| Checkpoint | `google/vit-base-patch16-224-in21k` | **RECOVERED** — the exact string in the historical code |
| Pretrained | yes | RECONSTRUCTED |
| Frozen | yes (default) | RECONSTRUCTED |
| Patch size | 16 | from checkpoint config |
| Embedding dim | 768 | from checkpoint config |
| Layers | 12 | from checkpoint config |
| Heads | 12 | from checkpoint config |
| MLP dim | 3072 | from checkpoint config |
| Input normalisation | mean = std = 0.5 | from checkpoint config |
| Output used | 196 patch tokens, CLS dropped, 1×1-projected to 3 channels, bilinear-upsampled | RECONSTRUCTED |

The historical code loaded **no transformer weights at all**, so no layer count,
head count or embedding dimension can be recovered from it — those come from the
named checkpoint's own config, not from any recovered hyperparameter. The ViT is
frozen because the manuscript never claims to fine-tune it, and because freezing
makes TF-RGB deterministic given the image, which allows the map to be
precomputed once for the whole dataset.

### 2.4 Fusion — exact operations

`src/aetfpe/fusion/ops.py`. With `F_PE, F_TF ∈ R^(B×3×224×224)`:

| Arm | Operation | Output | Extra params |
|---|---|---|---|
| F1 add | `F = ½(F_PE + F_TF)` | `[B,3,224,224]` | 0 |
| F2 concat | `F = [F_PE ; F_TF]` | `[B,6,224,224]` | 0 (**but the classifier stem widens to 6 channels**) |
| F3 linear | `F = σ(BN(W·[F_PE ; F_TF] + b))`, `W ∈ R^(3×6×1×1)` | `[B,3,224,224]` | 21 + BN |
| F4 attention | `s = σ(W₂·ReLU(W₁·GAP([F_PE ; F_TF])))`; `F = σ(BN(W·(s ⊙ [F_PE ; F_TF])))` | `[B,3,224,224]` | ~1 k |
| F5 AE | `F = Dec(Enc([F_PE ; F_TF]))` | `[B,3,224,224]` | 259,907 |

The auto-encoder always consumes the **concatenated** map, so AE fusion is a
genuine alternative to the operators above rather than a wrapper around one of
them. F4 substitutes a lightweight SE-style channel gate for full
cross-attention: at 196 tokens and image resolution, cross-attention costs more
than the auto-encoder it is meant to be a cheap foil for, which would make the
complexity comparison incoherent. The substitution is stated in the results table.

### 2.5 Auto-encoder — **RECONSTRUCTED**

`src/aetfpe/autoencoder/model.py`, `losses.py`

The manuscript calls it "stacked" (title), "sparse" (§3.3) and "a denoising
latent regularizer" (§1). Per the rule that terminology must follow the code:
the implementation uses **corrupted input with a clean reconstruction target**
(denoising), **plus a KL sparsity penalty** (sparse), across **three encoder and
three decoder stages** (stacked). The correct term, and the only one the revised
manuscript may use, is:

> **stacked sparse denoising auto-encoder**

| Property | Value |
|---|---|
| Input | `[B, 6, 224, 224]` (PE-RGB ⊕ TF-RGB), or `[B, 3, 224, 224]` for the RGB-AE arm |
| Reconstruction target | the **clean** RGB image, `[B, 3, 224, 224]` |
| Encoder | 3 × `[Conv3×3 s2 → BN → ReLU]`, 224→112→56→28, widths 32 → 64 → 128 |
| Latent | `[B, 128, 28, 28]` = **100,352 dims**, sigmoid activation |
| Decoder | 3 × `[ConvT4×4 s2 → BN → ReLU]`, 28→56→112→224, final sigmoid |
| Output | `[B, 3, 224, 224]` in `[0, 1]` |
| Reconstruction loss | `MSE(x̂, x_clean)` |
| Sparsity | `β · Σ_j KL(ρ ‖ ρ̂_j)` over the 128 latent channels, `ρ = 0.05`, `β = 1e-3` |
| L2 | via optimizer weight decay, `λ = 1e-3` |
| Optimizer | AdamW, lr 7.14e-4, cosine schedule with 3 warm-up epochs |
| Training schedule | **3 reconstruction-only warm-up epochs**, then joint with the classifier |
| Parameters | 259,043 (3-ch input) / 259,907 (6-ch input) |

Corrected objective (manuscript Eqs. 5–6, whose notation is broken):

```
Δ_sparse(W, b) = Δ(W, b) + β · Σ_{j=1..s₂} KL(ρ ‖ ρ̂_j)
Δ(W, b)        = (1/m) Σ_{i=1..m} ‖x̂⁽ⁱ⁾ − x⁽ⁱ⁾‖²  +  (λ/2) ‖W‖²
```

The latent uses a sigmoid so channel means `ρ̂_j` are valid Bernoulli parameters,
which is what makes the KL term well defined.

**Reconstruction decision — the warm-up stage.** The first smoke test exposed a
genuine deadlock: reconstruction MSE (~0.04) and cross-entropy (~3.8) differ by
two orders of magnitude, so the AE received almost no gradient pressure while the
classifier could not learn from an unreconstructed input. The model sat at chance
(0.0256 = 1/39) and the reconstruction loss did not improve. Two changes fix it,
both classical and both fixed a priori rather than tuned on results:

1. `ae_loss_weight = 10.0`, chosen so the two terms are comparable in magnitude
   at the start of joint training.
2. `ae_warmup_epochs = 3` — reconstruction-only epochs with no classification
   gradient. This is exactly the pretraining a *stacked* auto-encoder classically
   receives, so it strengthens rather than weakens the fidelity of the
   reconstruction. Checkpoints are never selected from warm-up epochs.

### 2.6 YOLOv8 interface — explicit

**YOLO receives a reconstructed RGB image.** Not a latent, not a projected
latent, not an internal feature injection.

| Arm | Tensor into the classifier | Stem modified? |
|---|---|---|
| A0, A1, M1, M2, M3, all baselines | `[B, 3, 224, 224]` | no |
| A2, A3, A4, A5, F1, F4 | `[B, 3, 224, 224]` | no |
| **F2 (plain concatenation)** | `[B, 6, 224, 224]` | **yes — first Conv2d widened 3→6** |

For F2 only, `adapt_stem()` widens the first `Conv2d`; existing weights are tiled
and rescaled by `1/reps` so the response to a duplicated input is unchanged and
that arm is not handed a worse initialisation than the others. This is the single
place any classifier is touched, and it is disclosed in the fusion table rather
than left for a reviewer to find. It is also the only arm for which the
manuscript's claim of integration "without requiring modifications to the
underlying classifiers" does not hold.

### 2.7 Training protocol — **RECONSTRUCTED, one protocol for all arms**

Values follow the *resolved* settings in `log-org-280223`, not the values the
manuscript claims:

| Setting | Value | Provenance |
|---|---|---|
| Image size | 224 | recovered |
| Batch size | 128 | recovered |
| Epochs | 30 | recovered (manuscript claims 50) |
| Optimizer | AdamW | recovered (`optimizer=auto` resolved to this) |
| Learning rate | 7.14e-4 | recovered (manuscript claims 0.01; the log prints `ignoring 'lr0=0.01'`) |
| Weight decay | 1e-3 | recovered |
| Warm-up | 3 epochs, then cosine | recovered / reconstructed |
| Seed | 0, `deterministic=True` | recovered |
| Augmentation | hflip 0.5, RandAugment(2, 9), RandomErasing 0.4 | reconstructed from the log's `auto_augment=randaugment, erasing=0.4, fliplr=0.5` |
| Checkpoint | best validation top-1 | reconstructed |
| Pretrained | yes, 156/158 items transferred | **recovered and reproduced exactly** |

**Reconstruction decision — custom training loop.** All arms train through one
PyTorch loop (`scripts/train.py`) wrapping
`ultralytics.nn.tasks.ClassificationModel`, rather than through the Ultralytics
trainer. Three reasons: it makes "identical experimental conditions"
(Reviewer #10.7) enforced by construction instead of asserted; it lets the
front-end run on the fly, avoiding ~8 GB of materialised PNGs per arm on a disk
with 14 GB free; and it permits joint AE + classifier optimisation, which the
Ultralytics trainer does not expose. Validation that this is faithful: the model
reports **1,488,247 parameters** and **156/158 transferred items** — both
identical to `log-org-280223`.

Augmentation is applied to the raw image **before** the front-end, so PE, TF and
AE all see a consistent view.

---

## 3. Corruption benchmark

`configs/corruptions.yaml`, `src/aetfpe/corruptions/`

**Training augmentation and robustness corruption are strictly separated.** The
default protocol is: train clean, validate clean, test on clean **plus** frozen
corrupted copies of the same test split. Training and validation splits are never
corrupted, except in arm M3, which exists specifically to test the
corruption-aware-training explanation and applies its augmentation in the
dataloader without touching the frozen files.

### 3.1 Legacy Types 1–3 — **RECONSTRUCTED**

No generation code survives. Two contradictions in the source prose had to be
resolved, and both resolutions are configurable:

| Type | Definition | Ambiguity resolved |
|---|---|---|
| **1 — pepper** | fraction `r` of pixels replaced; `salt_vs_pepper = 0.5`; whole pixels, not per-channel; `r ∈ {0.02, 0.10, 0.20, 0.30, 0.40, 0.50}` | §4.3.1 says "white and black dots" (salt-and-pepper) while §4.6 says "replaces pixels with black" (pepper only). Default is 50/50; `salt_vs_pepper: 0.0` reproduces the pepper-only reading |
| **2 — transparency** | `I_out = α·I_labelled + (1−α)·I_distractor`, `α = 0.7`; distractor drawn deterministically from a **different** class in the same split; label follows the foreground | §4.3.2 says the blend makes "the features of the image with higher transparency ... more prominent", which inverts itself. Default reads "70% transparency" as: the labelled image is composited at α=0.7 and dominates |
| **3 — combined** | pepper applied to the labelled image **first**, then composited with a clean distractor | §4.3.3 states this order explicitly |

These are **reconstructions, not historical equivalents.** Any comparison against
the original Tables 2–3 must say so.

### 3.2 New benchmark — Reviewer #11

| Family | easy | medium | hard |
|---|---|---|---|
| Gaussian noise (σ, 0–255) | 10 | 25 | 50 |
| Gaussian blur (radius, px) | 1.0 | 2.0 | 4.0 |
| Brightness (multiplier) | 0.7 | 0.5 | 0.3 |
| JPEG (quality) | 40 | 20 | 10 |

Fixed **before** any model was evaluated. Motion blur and contrast are
implemented but excluded from the default plan: they add rows without adding an
argument.

### 3.3 Determinism

The RNG for each image is derived from `(seed, relative_path, corruption,
severity)` via blake2b — not from iteration order, worker count, or Python's
salted `hash()`. So a corrupted file is reproducible in isolation.

`corruption_manifest.csv` records `original_path, corrupted_path, class,
corruption, severity, seed, parameters, checksum` for every file. `evaluate.py`
records the manifest's own sha256 in each result, so two runs can be proven to
have been scored on the same bytes.

**Verified:** 2,028 files regenerated and compared against their recorded
sha256 — **0 mismatches**.

This directly repairs the defect in the inherited TTA spec, where every transform
fired on an unseeded per-image `p=` draw, so each model was scored on a
*different* corrupted test set.

---

## 4. Deduplicated run matrix

`python scripts/print_run_matrix.py`

| Run | Group | PE | TF | AE | Fusion | Extra | Reused as |
|---|---|---|---|---|---|---|---|
| M1_legacy_lut | mechanism | – | – | – | – | legacy LUT (0.2/0.8) | |
| M2_photometric | mechanism | – | – | – | – | gamma 1.6 | |
| M3_aug_control | mechanism | – | – | – | – | corruption-augmented training | |
| A0_baseline_rgb | ablation | – | – | – | – | | |
| A1_pe_only | ablation | Y | – | – | – | | |
| A2_tf_only | ablation | – | Y | – | linear | | |
| A3_pe_tf_no_ae | ablation | Y | Y | – | linear | | **F3_fusion_linear** |
| A4_rgb_ae | ablation | – | – | Y | – | | |
| A5_aetfpe_full | ablation | Y | Y | Y | linear | | **F5_fusion_ae** |
| F1_fusion_add | fusion | Y | Y | – | add | | |
| F2_fusion_concat | fusion | Y | Y | – | concat | 6-ch stem | |
| F4_fusion_attention | fusion | Y | Y | – | attention | | |
| B1_resnet50 | baseline | – | – | – | – | ResNet-50 | |
| B2_efficientnet_b0 | baseline | – | – | – | – | EfficientNet-B0 | |
| B3_vit_b16 | baseline | – | – | – | – | ViT-B/16 | |

**Core ablation + fusion = 9 unique training runs** (F3 and F5 are reused, not
retrained), within the 8–10 target. Plus 3 mechanism controls and 3 fair
baselines = **15 total**.

### 4.1 Why the mechanism controls exist, and why they run first

Recomputing the archived YOLOv7 pair gives **9.08 % → 16.40 %, a factor of
1.81**, against the manuscript abstract's claimed "approximately 1.9×". The
robustness effect is therefore probably real and reproducible — but the transform
that produced it is a zero-parameter pointwise lookup table.

So the open question is no longer "does AE-TFPE help?" It is **"does the
auto-encoder beat the trivial thing that already produced the gain?"** M1 (the
LUT itself), M2 (a monotonic photometric control with matched deviation but no
wrap discontinuities) and M3 (plain augmentation) are the three competing
explanations. Running them first, alongside A0 and A5, reaches a decision point
in a few days instead of after the full matrix.

If A5 does not clearly beat M1, M2 and M3, no component ablation can rescue it,
and the honest revision reports the mechanism that does work.

---

## 5. Result provenance

Each run directory contains:

```
results/<group>/<name>/
  config.yaml            resolved config, including everything inherited from _base
  environment.json       git commit + dirty flag, python/torch versions, platform,
                         device, model description, class list, dataset fingerprints
  metrics.csv            per-epoch loss / top-1 / lr / stage / seconds
  checkpoint.pt          best-val-top1 weights + config + class list
  train_summary.json     protocol, best val top-1, wall clock
  test_clean.json/.csv   clean test metrics, per class, plus confusion matrix
  test_corruptions.csv   one row per corruption × severity
  eval_summary.json      checkpoint sha256 + corruption manifest sha256
  per_class/<c>_<s>.json full per-class breakdown per condition
  analysis/              confusion matrices and per-class tables
  latent/                drift statistics and t-SNE/UMAP figures
```

Dataset fingerprints hash the sorted per-class file listing, so a silently
changed split is detectable. No manuscript number should ever be typed by hand;
all of them are derivable from these CSVs.

---

## 6. Historical result policy

Historical results are **evidence about the past**, not results of the new
protocol, and the two are never combined in one table. Specifically:

- Tables 2–3 and Figs. 7–10 cannot be reproduced: the corrupted datasets are
  gone, the generation code never existed, and the evaluation was
  detection-based with an undisclosed confidence threshold.
- §4.6.2's YOLOv7 baseline (9.08 %) is the one figure that traces exactly. Its
  paired proposed figure (14.8 %) does not.
- The revised manuscript should report the new frozen protocol as its primary
  evidence, and either drop the historical tables or mark them explicitly as
  non-reproducible legacy measurements.

---

## 7. Blueberry dataset — Phase 12

**Partially recoverable, but not usable as published.**
`classified_dataset/` contains **74 images** across 15 class folders, **6 of them
empty**, with Roboflow-export filenames carrying `_train`/`_test` suffixes:

| Class | Images | | Class | Images |
|---|---|---|---|---|
| Twigs_defolated_by_leaf_spots | 20 | | Healthy_fall_leaves | 10 |
| Blueberry_rust_lesions | 14 | | Abiotic_symptoms | 5 |
| Exobasidium_green_spot | 11 | | Large_spots | 3 |
| Early-stage_powdery_mildew | 10 | | Late-season_powdery_mildew | 1 |
| Anthracnose_leaf_spot, Blueberry_rust_symptoms, Double_spot_lesions, Early_season_powdery_mildew_infection, Exobasidium, Exobasidium_leaf_and_fruit_spot, Septoria_leaf_spots | 0 each | | | |

74 images across 8 usable classes, one of which has a single image, with no
recorded split, no protocol and no provenance documentation, cannot support
§4.6.4 or Figs. 11–12. The recommendation stands: **withdraw those results and
state the limitation**, while noting in the response letter that a small
expert-annotated set exists but is too small and too incompletely documented for
a quantitative claim. Do not fabricate replacement field results.

---

## 8. Known gaps

| Gap | Status |
|---|---|
| Original PE, transformer and AE implementations | Never existed. Reconstructed here, labelled as such |
| Original corruption generation | Not recoverable. Reconstructed, labelled as such |
| Which blend ratio (0.1/0.9 vs 0.2/0.8) backs which table | Not recoverable |
| §4.6.2's 14.8 % figure | Does not reproduce from any surviving artifact |
| Exact historical train/val/test split membership | The 55,259-image split exists on disk and matches the log exactly, but the script that built it does not survive |
| `Plant_Handled-*` datasets | Result directories survive; the datasets do not |
| Whether reported numbers are val or test | Historical evidence points to `val`; needs an explicit statement in the revision |
