# Frozen Corruption Specification

**Frozen 15 August 2026, before any final model result was observed.**

These parameters are the contract. They must not be changed after seeing
model results — doing so would make every robustness comparison post-hoc. The
file of record is `configs/corruptions.yaml`; this document explains and
classifies it.

Provenance categories:

- **RECOVERED** — the value itself survives in a historical artifact.
- **RECONSTRUCTED** — the family existed historically but its definition does not survive; parameters here are a documented rebuild.
- **NEW** — introduced by this revision at Reviewer #11's request.

---

## 1. The table

| Corruption | Severity | Exact parameters | Seed behaviour | Provenance | Motivation |
|---|---|---|---|---|---|
| **clean** | – | identity; resize 224 only | none needed | NEW (reference) | Reviewer #11 asked for complete clean results |
| **pepper** | 002 | `ratio=0.02, salt_vs_pepper=0.5, per_channel=false` | per-image, `blake2b(seed, rel_path, "pepper", "002")` | ratio **RECOVERED**, definition **RECONSTRUCTED** | Manuscript Table 2 severity |
| | 010 | `ratio=0.10, salt_vs_pepper=0.5, per_channel=false` | same | same | " |
| | 020 | `ratio=0.20, salt_vs_pepper=0.5, per_channel=false` | same | same | " |
| | 030 | `ratio=0.30, salt_vs_pepper=0.5, per_channel=false` | same | same | " |
| | 040 | `ratio=0.40, salt_vs_pepper=0.5, per_channel=false` | same | same | " |
| | 050 | `ratio=0.50, salt_vs_pepper=0.5, per_channel=false` | same | same | Headline severity in the abstract |
| **transparency** | 70 | `alpha=0.7`; `I = 0.7·I_label + 0.3·I_distractor` | distractor chosen by `blake2b(seed, rel_path, "transparency", "70", "distractor")`, constrained to a different class | family **RECOVERED**, semantics **RECONSTRUCTED** | Manuscript Type 2 |
| **pepper_transparency** | 002–050 | `ratio` as above, then `alpha=0.7`, `salt_vs_pepper=0.5`; noise applied to the labelled image **first** | both draws seeded as above | order **RECONSTRUCTED** from §4.3.3 prose; parameters **RECONSTRUCTED** | Manuscript Type 3 / Table 3 |
| **gaussian_noise** | easy | `sigma=10` (0–255 units) | per-image seeded | **NEW** | Tests whether the gain is impulse-specific or general |
| | medium | `sigma=25` | | | |
| | hard | `sigma=50` | | | |
| **gaussian_blur** | easy | `sigma=1.0` px (PIL `GaussianBlur` radius) | deterministic, no RNG | **NEW** | A low-pass failure mode, opposite in character to noise |
| | medium | `sigma=2.0` px | | | |
| | hard | `sigma=4.0` px | | | |
| **brightness** | easy | `factor=0.7` (multiplicative) | deterministic, no RNG | **NEW** | The paper's own motivation cites illumination fluctuation; this is where a contrast-remap mechanism should show its hand |
| | medium | `factor=0.5` | | | |
| | hard | `factor=0.3` | | | |
| **jpeg** | easy | `quality=40` (libjpeg) | deterministic, no RNG | **NEW** | Realistic for field capture; cheap to generate |
| | medium | `quality=20` | | | |
| | hard | `quality=10` | | | |

**Totals:** 21 configurations for the frozen benchmark
(1 clean + 6 pepper + 1 transparency + 6 pepper_transparency + 4 families × ...
wait, 4 new families × 3 = 12; 1+6+1+6+12 = **26**). The 21 figure quoted
elsewhere excluded five of the `pepper_transparency` ratios; **26 is correct**
and is what `expand_plan()` produces.

Implemented but **excluded by default**: `contrast`, `motion_blur`. They add rows
without adding an argument. Their exclusion is deliberate and should be stated in
the paper.

---

## 1a. Reproducibility: pixel content vs encoded bytes

Two different claims, and only one of them is guaranteed.

| | Guaranteed? | Why |
|---|---|---|
| **Pixel-array reproducibility** — the corruption arithmetic yields identical numbers | **Yes**, for 25 of 26 configurations | Pure numpy float64 with an explicit kernel; `numpy.random.Generator(PCG64)` is stream-stable across versions and platforms by NumPy's own compatibility policy |
| **Encoded-file byte reproducibility** — the PNG bytes are identical | **No** | PNG bytes depend on the zlib build and encoder settings. Measured locally: the same pixels give four different file hashes under `optimize=False/True` and `compress_level=1/9`, at identical file length |

Consequences, implemented:

- **`pixel_sha256` is the manifest's field of record.** `file_sha256` is retained
  but informational. `--verify` fails only on pixel mismatch; a file-only
  mismatch prints a note saying the image encoder changed and the benchmark is
  intact.
- **`Image.resize` is called with an explicit `BICUBIC`.** The historical code
  relied on the default, which Pillow resolves to BICUBIC in 10.2.0 — but a
  version-dependent default is not something a frozen benchmark may rest on.
  Byte-exact reproduction of the legacy dataset was re-verified after this
  change: **21/21 exact matches, max MAE 0.000000.**
- **Gaussian blur no longer uses `PIL.ImageFilter.GaussianBlur`.** Pillow's blur
  implementation has changed between releases, which would have made that
  corruption's pixel content Pillow-dependent. It is now a separable numpy
  float64 convolution with an explicit 3σ-truncated kernel and edge padding.

### The one genuine exception: JPEG

`jpeg` round-trips through libjpeg. libjpeg-turbo and reference libjpeg produce
different quantised output, and turbo releases have changed their DCT paths. Its
**pixel array**, not merely its file bytes, can differ between environments.

Mitigation rather than pretence:

1. `pillow==10.2.0` is pinned in `requirements.txt`.
2. The JPEG codec version is recorded in `generation_environment.json`.
3. `scripts/verify_reproducibility.py` runs every family on a fixed synthetic
   image and compares pixel hashes against `docs/reproducibility_reference.json`.
   Run it on Colab **before** Stage 2. It classifies any divergence as
   JPEG-only (pin or drop the family) versus numpy-level (stop immediately).

Reference environment: numpy 1.26.3, Pillow 10.2.0, libjpeg API 6.2, zlib 1.3.2,
arm64 macOS. Self-check: **26/26 configurations reproduce**.

---

## 2. Seeding contract

```python
rng = numpy.random.default_rng(
    int.from_bytes(blake2b(f"{seed}|{rel_path}|{corruption}|{severity}").digest(8), "big") % 2**32
)
```

- Derived from content, **not** from iteration order, worker count, or Python's
  salted `hash()` — so a single corrupted file is reproducible in isolation and
  is byte-identical on macOS and Linux.
- Deterministic corruptions (blur, brightness, JPEG, contrast, motion blur) take
  the RNG for signature uniformity but do not consume it.
- The distractor image for transparency corruptions is chosen from a separate
  derived stream, so changing the pepper ratio does not change which distractor
  is used.

**Verified:** 2,028 files regenerated and compared against recorded sha256 —
**0 mismatches**.

---

## 3. Which split gets corrupted

| Set | Split | Path | Generated at | Purpose |
|---|---|---|---|---|
| **Calibration (Stage 2A)** | `val` | `${OUTPUT_ROOT}/corruptions_val` | **Before** any training | AE hyperparameter calibration (V1), the G5 mechanism gate, all model-selection decisions |
| **Frozen benchmark (Stage 2B)** | `test` | `${OUTPUT_ROOT}/corruptions` | **After** every model and hyperparameter decision is locked | Final evaluation only, once per model, at Stage 8 |

Generating the test benchmark late is not merely tidy: it removes the
possibility of using it. **The test corruption benchmark must never inform model
selection**, and the cheapest guarantee is for it not to exist until selection is
over.

### 3a. Persistence policy — the pixels are ephemeral

The corrupted PNGs are ~21 GB and fully regenerable. They are **not** archived.
What is archived is a bundle of a few tens of MB that is sufficient to regenerate
and verify the benchmark from nothing:

| Persistent artifact | Content |
|---|---|
| `corruption_manifest.csv` | one row per file: source path, output path, class, corruption, severity, seed, exact parameters, **`pixel_sha256`**, `file_sha256` |
| `clean_split_manifest.csv` | the exact clean source images the benchmark was built from, in order |
| `generation_environment.json` | generator version, git commit + dirty flag, seed, image size, resample mode, configuration list, sha256 of `configs/corruptions.yaml`, sha256 of both manifests, and the full generation environment (python, platform, machine, numpy, Pillow, zlib, JPEG codec) |
| `configs/corruptions.yaml` | the parameter source, hashed into the bundle |
| `docs/reproducibility_reference.json` | per-family pixel hashes on a fixed synthetic image, for cross-platform checking |

| Ephemeral artifact | Content |
|---|---|
| the corrupted PNG tree | 216,710 files, ~21 GB, regenerable byte-identically at the pixel level |

Recovery after a runtime reset is: re-run the generator with the same seed, then
`--verify`. If pixel hashes match the archived manifest, the benchmark is
restored exactly. If they do not, the environment changed and the bundle says
precisely how.

Training and validation image splits are never corrupted on disk. Arm `M3` and
the AE's denoising objective apply corruption **in the dataloader**, at random
severities drawn per batch, and never touch these directories. That keeps the
training-time noise process disjoint from the frozen benchmark.

---

## 4. Severity calibration and why it is already frozen

The severities were fixed a priori: standard values from the corruption-robustness
literature for the new families, and the manuscript's own six ratios for pepper.
They were **not** tuned against any model.

A sanity check was run on a 2-epoch smoke model and showed monotone degradation
within every family, pepper hardest, transparency mildest. That check read
numbers from the corrupted **test** split — an exposure declared in
`PROVENANCE_MATRIX.md` §2.1. No parameter was changed as a result, and the
corrective action (a `val`-split calibration set) is now implemented. Any future
calibration uses `corruptions_val`.

---

## 5. What must be said in the manuscript

1. Types 1–3 in the revised paper are **reconstructions**. The original
   generation code does not survive, and two of its definitions were ambiguous in
   the source text: §4.3.1 describes Type 1 as "white and black dots" while §4.6
   describes it as pepper-only, and §4.3.2's transparency wording inverts itself.
   The choices made here are configurable and documented.
2. Results under reconstructed Types 1–3 are **not** numerically comparable to
   the original Tables 2–3, and no ratio should be computed across the two.
3. The new families are additions requested in review, with severities fixed
   before evaluation.
4. The benchmark is generated once, seeded, checksummed, and reused byte-identically
   by every model — which is precisely the property the historical evaluation
   lacked, since its Albumentations protocol re-drew every transform per model
   with no seed.
