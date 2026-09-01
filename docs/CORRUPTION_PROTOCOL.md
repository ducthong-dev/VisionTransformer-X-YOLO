# Evaluation Distributions — Recovered Construction Protocol

**Date:** 2 September 2026 · **Status:** recovered from source, verified against the
image data · **Evidence:** `results/eval_integrity/distribution_verification.json`

Written **before any model was evaluated on these sets**, so that what the tiers *are*
is fixed independently of what any model scores on them.

---

## 0. Summary for the impatient

The four evaluation distributions are **not** the frozen corruption benchmark defined in
`configs/corruptions.yaml`. That benchmark has never been generated. The four sets in
the dataset were produced by the sibling ResCBAM project's `augment_pt.py` using a
**single random draw from a mixed geometric + photometric + noise augmentation
pipeline**, with **no seed**.

They are usable, and their severity ordering is verified monotone. But they must be
described as **synthetic augmentation robustness**, never as "real-world robustness" and
never as a graded corruption benchmark with controlled severities.

---

## 1. Provenance

Generator: `Research 🍀/FPT/YOLOv8-ResCBAM/Fracture_Detection_Improved_YOLOv8/augment_pt.py`
with transforms from `dataset_loader.py`.

```python
for idx in range(len(test_dataset)):
    image, label = test_dataset[idx]           # LeafDataset(test_dir, transform=test_aug)
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * [0.229,0.224,0.225] + [0.485,0.456,0.406]   # denormalise
    img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(f"{save_dir}/{class_name}/img_{idx}.png")
```

`LeafDataset` sorts **class directories** but iterates files with a bare `os.listdir`,
so `idx` is a **global** index over classes-in-sorted-order, files-in-filesystem-order.

## 2. What each tier is — and what is not recoverable

`augment_pt.py` hard-codes one `save_dir` and one pipeline. The three tiers were produced
by editing `test_aug` between runs. **Only the last edit survives in the repository.**

| Tier | Directory | Pipeline parameters |
|---|---|---|
| Clean | `test` | original JPEG images, untouched |
| Easy | `augmented_test_images_easy` | **NOT RECOVERABLE** — overwritten |
| Moderate | `augmented_test_images_enhanced` | **NOT RECOVERABLE** — overwritten |
| Hard | `augmented_test_images_hardest` | the surviving `test_aug` (below), *presumed* |

The surviving `test_aug`, applied after `A.Resize(224,224)`:

| Op | Parameters | p |
|---|---|---|
| HorizontalFlip | — | 0.6 |
| VerticalFlip | — | 0.4 |
| RandomRotate90 | — | 0.5 |
| Rotate | limit 30° | 0.5 |
| RandomBrightnessContrast | ±0.25 / ±0.25 | 0.4 |
| ColorJitter | b/c/s 0.2, hue 0.15 | 0.4 |
| GaussNoise | var 10–60 | 0.4 |
| Blur | limit 5 | 0.3 |
| MotionBlur | limit 7 | 0.2 |
| MedianBlur | limit 5 | 0.2 |
| CoarseDropout | 1–4 holes, 16–48 px, fill 0 | 0.4 |
| ISONoise | shift .01–.05, int .1–.5 | 0.2 |
| Solarize | threshold 128 | 0.1 |
| Posterize | 4 bits | 0.1 |
| RandomFog | coef 0.1–0.3 | 0.15 |
| RandomRain | blur 2, drop len 20 | 0.15 |
| RandomShadow | lower half, 1–2 shadows | 0.15 |

Even for Hard this attribution is **presumed, not proven** — the file may have been
edited after that tier was written.

## 3. Reproducibility — the sets are not regenerable

`augment_pt.py` sets **no random seed** (no `random.seed`, `np.random.seed`, or
albumentations seed). Re-running it produces different images. **The saved PNGs are the
only record of the realised corruption**, and must be treated as frozen primary data:
never regenerate, never overwrite.

## 4. Verified properties — **[MEASURED]**

`scripts/verify_eval_distributions.py`, run 2026-09-02. STATUS: **PASS**.

**Structure.** All four sets: 8335 images, 39 classes, identical class ordering,
identical per-class supports. Clean listing sha `aad05ffc…`; all three augmented sets
share listing sha `1a447413…` (identical *filenames*, by construction).

**Distinctness.** Per-file content hashes: 8335 unique in each set. Pairwise
byte-identical files — easy/moderate 125 (1.50 %), easy/hard 6 (0.07 %),
moderate/hard 3 (0.04 %). These are images where no stochastic op happened to fire; the
rate falls with severity exactly as a probabilistic pipeline predicts. **The three tiers
are distinct sets, not copies.**

**Severity is monotone.** Measured against the reconstructed no-op render of each
image's own clean source:

| Tier | untouched fraction | mean abs deviation | median abs deviation |
|---|---|---|---|
| Easy | 25.67 % | 28.375 | 26.030 |
| Moderate | 3.33 % | 44.502 | 44.664 |
| Hard | 0.67 % | 52.790 | 52.685 |

Ordering easy < moderate < hard holds on every statistic.

**A large share of the difference is geometric, not corruption.** Comparing each
augmented image against all eight dihedral transforms of its clean source:

| Tier | identity fits best (no flip/rot) | a flip/rot fits ≥40 % better |
|---|---|---|
| Easy | 69.9 % | 22.6 % |
| Moderate | 27.4 % | 40.7 % |
| Hard | **15.9 %** | 26.1 % |

**In the Hard tier roughly 84 % of images have been flipped or rotated.** Flips and
90° rotations are *pose changes*, not corruptions, and a denoising auto-encoder is not
expected to undo them.

## 5. Consequences for what may be claimed

1. **Call it "synthetic augmentation robustness."** Not "real-world robustness", not a
   corruption benchmark with severity levels. The transformations do not justify either.
2. **Noise resilience cannot be isolated on these sets.** Geometric invariance,
   photometric shift, occlusion (CoarseDropout, fog, rain, shadow) and additive noise all
   vary together and were never separated. Reviewer #10's "noise-resilient latent
   features" challenge is **not** answered by these four sets alone.
3. **Easy and Moderate are empirically graded but not specified.** Report the measured
   severity statistics in §4 as the definition of the tiers, since the generating
   parameters are gone.
4. **No per-severity attribution.** No claim of the form "robust to blur at severity k".
5. **The properly frozen alternative exists and is unused.** `configs/corruptions.yaml`
   + `scripts/generate_corruptions.py` define seeded, parameterised, manifest-backed
   corruption families with pixel-level hashes. If a reviewer demands a controlled
   corruption benchmark, that is the instrument to run — on the test split, only after
   all model decisions are locked.

## 6. Clean ↔ augmented sample mapping — **PROVEN**

Recovered rule, frozen to `results/eval_integrity/clean_augmented_mapping.json`:

```
img_<GLOBAL>.png  in class c   <->   os.listdir(test/c)[GLOBAL - class_start(c)]
classes sorted; class_start = cumulative per-class counts in sorted-class order
```

Verification: 300 probed images of Easy — **77 exact pixel matches** at the predicted
clean file (`maxdiff <= 2`; a wrong pair differs by >100), 223 where an augmentation op
fired so no exact match can exist, **0 contradictions**, and **77/77 negative controls
passed** (a different image of the same class never matched).

Because the rule depends on filesystem ordering, the mapping is **materialised to disk**
and all downstream analysis reads that frozen file rather than calling `os.listdir`
again. Two independent local dataset copies were checked and return identical order.

**Consequence:** cross-distribution image-level paired analysis (same source image,
clean vs Easy/Moderate/Hard) **is available** and may be used.

## 7. Known confounds to carry into the claim audit

- **PNG vs JPEG.** Clean is original JPEG at native resolution; the augmented sets are
  224×224 PNG that additionally passed a float normalise/denormalise round trip with
  uint8 truncation. Every clean→corrupted delta therefore contains a resize and a
  recompression component.
- **Resize is in the augmented path only.** Clean images are resized by the evaluation
  dataloader instead, which may use different interpolation.
- **Geometric transforms dominate the harder tiers** (§4).
- **Single realisation.** One random draw per image per tier; no repetition, so
  corruption-sampling variance is unmeasurable.

---

## 8. Consequence: a controlled benchmark was added alongside these sets

§4 and §5 establish that these four distributions cannot isolate noise robustness —
~84 % of `hard` is flipped or rotated, and geometry, photometry, occlusion and noise
vary together.

**These sets are not discarded or modified.** They were part of the planned campaign,
their severity ordering is verified monotone, and they remain a valid measurement of
**synthetic augmentation robustness**.

A second, separately labelled benchmark was added on 2026-09-02 to answer the
reviewers' noise challenge without the geometric confound:

> **Controlled Synthetic Corruption Benchmark** — 6 deterministic, label-preserving,
> **non-geometric** families (Gaussian noise, impulse noise, Gaussian blur, brightness,
> contrast, JPEG) × mild / moderate / severe, on the clean test split only.

Specification: `configs/controlled_corruptions.yaml` (frozen, hashed).
Freeze + manifest: `scripts/generate_controlled_corruptions.py` →
`results/controlled_corruptions/`. Evaluation:
`scripts/evaluate_controlled_corruptions.py` → `results/evaluation_controlled/`,
a separate output tree from `results/evaluation/`.

It was introduced **after** this audit revealed the geometric confounding, but **before**
any model's test performance was inspected. It is a frozen specification, **not a
pre-registration**. Rationale, parameters, seed rule, model set and statistics:
`docs/GATE1_RESOLUTION_AND_ANALYSIS_PROTOCOL.md` §6.

**The two benchmarks are reported separately and are never merged into a single
unexplained average.**

---

## 9. Incidental data-quality finding — a duplicate pair in the clean test split

Freezing the controlled benchmark surfaced 12 duplicate pixel hashes among 150,030
corrupted samples. All 12 are the *same pair of source images* across the four
**deterministic** families (blur, brightness, contrast, JPEG × 3 severities = 12); the
two stochastic families differ because their seeds are derived from the file path.

```
test/Apple___healthy/image (1120).JPG
test/Apple___healthy/image (163).JPG
        both 11,624 bytes, file sha256 6b242d89b134aa87…, pixel-identical
```

These are **byte-identical duplicate files**. The clean test split therefore contains
8,335 files but 8,334 distinct images.

Impact is negligible and unbiased: every model sees the duplicate equally, so no
model-vs-model comparison is affected, and 1 duplicate in 8,335 (0.012 %) cannot move
an accuracy figure meaningfully. **The split is not modified** — the frozen 8,335-image
listing is the authoritative test set every training run and both benchmarks refer to.
It is recorded here so the count is not later mistaken for an error.

The finding also confirms the integrity machinery works: deterministic corruptions
produced identical output for identical input, exactly as specified.
