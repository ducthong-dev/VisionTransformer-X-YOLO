# Decoder Shape-Path Audit — C2-7 / C2-14 / C2-28

**Date:** 16 August 2026 · **Status:** documentation audit only. Nothing
redesigned, nothing removed, nothing optimised.
**Reproduce:** `python scripts/audit_decoder_path.py` (CPU, offline, no dataset)

Commissioned because `t4_benchmark.json` metadata appeared internally
inconsistent: C2-28 reports `decoder = "2 x ConvT4x4 s2, 28 -> 224"`, yet two
stride-2 stages reach only 112, while `classifier_input` is correctly
`[B, 3, 224, 224]`.

**Verdict: the metadata string is wrong; the architecture is correct.** There is
**no** hidden interpolation in any decoder. See `PROTOCOL_AMENDMENT_2026-08-16.md`
§A3.

Shapes and MAC counts are hardware-independent, so this audit is valid from any
device. It was run on CPU. No latency figure here is from a Mac.

---

## A. Resolution of the apparent inconsistency

`SlimFeatureSpaceAE.describe()` computes the stage count as
`len(self.decoder) // 2`. `self.decoder` is
`[_up_block × len(decoder_widths)] + [ConvTranspose2d] + [Sigmoid]`, so that
expression counts nothing meaningful. The true count is `len(decoder_widths) + 1`.

| Candidate | Reported string | **Actual stages** | **Actual spatial path** |
|---|---|---|---|
| C2-7 | "3 x [ConvT4x4 s2], 7 -> 224" | **5** | 7 → 14 → 28 → 56 → 112 → 224 |
| C2-14 | "2 x [ConvT4x4 s2], 14 -> 224" | **4** | 14 → 28 → 56 → 112 → 224 |
| C2-28 | "2 x [ConvT4x4 s2], 28 -> 224" | **3** | 28 → 56 → 112 → 224 |

Every variant reaches 224 by learned transposed convolution alone. The
constructor already enforces this: it raises unless
`n_up == len(decoder_widths) + 1`, where `n_up` is the number of doublings from
`grid` to `out_size`. The guard is correct; only the description is not.

---

## B. Layer-by-layer trace — **[MEASURED]**

Every shape-changing operation from input image to classifier input. Implicit
resizes are caught by patching the functional API (`F.interpolate`,
`F.adaptive_avg_pool2d`, `F.upsample`), so an operation that is not a registered
module still appears. Rows marked **✱** are implicit resizes.

MACs: `Conv2d` = `Cout·Hout·Wout·(Cin/groups)·k²` · `ConvTranspose2d` =
`Cin·Hin·Win·(Cout/groups)·k²` (input-scatter; see amendment §A2). Elementwise
layers carry no MACs.

### C2-28 — grid 28×28, AE input 51 ch (48 backbone + 3 pooled PE)

| # | Layer | Operation | k | s | p | Input | Output | MMACs |
|---|---|---|---|---|---|---|---|---|
| 1 ✱ | *(frontend)* | `adaptive_avg_pool2d` | – | – | – | 1×3×224×224 | 1×3×28×28 | – |
| 2 | `ae.encoder.0` | Conv2d | 1×1 | 1×1 | 0 | 1×51×28×28 | 1×64×28×28 | 2.559 |
| 3 | `ae.encoder.1` | BatchNorm2d | – | – | – | 1×64×28×28 | 1×64×28×28 | – |
| 4 | `ae.encoder.2` | Sigmoid | – | – | – | 1×64×28×28 | 1×64×28×28 | – |
| 5 | `ae.decoder.0.0` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×64×28×28 | 1×48×56×56 | 38.535 |
| 6 | `ae.decoder.0.1` | BatchNorm2d | – | – | – | 1×48×56×56 | 1×48×56×56 | – |
| 7 | `ae.decoder.0.2` | ReLU | – | – | – | 1×48×56×56 | 1×48×56×56 | – |
| 8 | `ae.decoder.1.0` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×48×56×56 | 1×32×112×112 | 77.070 |
| 9 | `ae.decoder.1.1` | BatchNorm2d | – | – | – | 1×32×112×112 | 1×32×112×112 | – |
| 10 | `ae.decoder.1.2` | ReLU | – | – | – | 1×32×112×112 | 1×32×112×112 | – |
| 11 | `ae.decoder.2` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×32×112×112 | 1×3×224×224 | 19.268 |
| 12 | `ae.decoder.3` | Sigmoid | – | – | – | 1×3×224×224 | 1×3×224×224 | – |

**3 transposed stages. AE total 137.432 MMACs = 0.2749 GFLOPs.**
latent `[1,64,28,28]` → classifier input `[1,3,224,224]` → logits `[1,39]`.

### C2-14 — grid 14×14, AE input 67 ch (64 + 3)

| # | Layer | Operation | k | s | p | Input | Output | MMACs |
|---|---|---|---|---|---|---|---|---|
| 1 ✱ | *(frontend)* | `adaptive_avg_pool2d` | – | – | – | 1×3×224×224 | 1×3×14×14 | – |
| 2 | `ae.encoder.0` | Conv2d | 1×1 | 1×1 | 0 | 1×67×14×14 | 1×64×14×14 | 0.840 |
| 3–4 | `ae.encoder.1–2` | BatchNorm2d, Sigmoid | – | – | – | 1×64×14×14 | 1×64×14×14 | – |
| 5 | `ae.decoder.0.0` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×64×14×14 | 1×48×28×28 | 9.634 |
| 6–7 | `ae.decoder.0.1–2` | BatchNorm2d, ReLU | – | – | – | 1×48×28×28 | 1×48×28×28 | – |
| 8 | `ae.decoder.1.0` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×48×28×28 | 1×32×56×56 | 19.268 |
| 9–10 | `ae.decoder.1.1–2` | BatchNorm2d, ReLU | – | – | – | 1×32×56×56 | 1×32×56×56 | – |
| 11 | `ae.decoder.2.0` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×32×56×56 | 1×16×112×112 | 25.690 |
| 12–13 | `ae.decoder.2.1–2` | BatchNorm2d, ReLU | – | – | – | 1×16×112×112 | 1×16×112×112 | – |
| 14 | `ae.decoder.3` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×16×112×112 | 1×3×224×224 | 9.634 |
| 15 | `ae.decoder.4` | Sigmoid | – | – | – | 1×3×224×224 | 1×3×224×224 | – |

**4 transposed stages. AE total 65.066 MMACs = 0.1301 GFLOPs.**

### C2-7 — grid 7×7, AE input 323 ch (320 + 3)

| # | Layer | Operation | k | s | p | Input | Output | MMACs |
|---|---|---|---|---|---|---|---|---|
| 1 ✱ | *(inside MobileViT stage 4)* | `interpolate(bilinear)` | – | – | – | 1×96×7×7 | 1×96×**8×8** | – |
| 2 ✱ | *(inside MobileViT stage 4)* | `interpolate(bilinear)` | – | – | – | 1×96×8×8 | 1×96×**7×7** | – |
| 3 ✱ | *(frontend)* | `adaptive_avg_pool2d` | – | – | – | 1×3×224×224 | 1×3×7×7 | – |
| 4 | `ae.encoder.0` | Conv2d | 1×1 | 1×1 | 0 | 1×323×7×7 | 1×64×7×7 | 1.013 |
| 5–6 | `ae.encoder.1–2` | BatchNorm2d, Sigmoid | – | – | – | 1×64×7×7 | 1×64×7×7 | – |
| 7 | `ae.decoder.0.0` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×64×7×7 | 1×48×14×14 | 2.408 |
| 10 | `ae.decoder.1.0` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×48×14×14 | 1×32×28×28 | 4.817 |
| 13 | `ae.decoder.2.0` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×32×28×28 | 1×16×56×56 | 6.423 |
| 16 | `ae.decoder.3.0` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×16×56×56 | 1×8×112×112 | 6.423 |
| 19 | `ae.decoder.4` | **ConvTranspose2d** | 4×4 | 2×2 | 1 | 1×8×112×112 | 1×3×224×224 | 4.817 |
| 20 | `ae.decoder.5` | Sigmoid | – | – | – | 1×3×224×224 | 1×3×224×224 | – |

*(BatchNorm2d + ReLU follow each of rows 7/10/13/16; omitted for width.)*

**5 transposed stages. AE total 25.900 MMACs = 0.0518 GFLOPs.**

---

## C. Every implicit resize, and what it costs

| Candidate | Implicit resizes in the front-end path | Where |
|---|---|---|
| C2-28 | **1** — `adaptive_avg_pool2d` 3×224×224 → 3×28×28 | front-end, PE branch |
| C2-14 | **1** — `adaptive_avg_pool2d` 3×224×224 → 3×14×14 | front-end, PE branch |
| C2-7 | **3** — the same pool, **plus two bilinear interpolations inside MobileViT stage 4** | backbone-internal |

Three findings:

**1. There is no resize in any decoder.** Not bilinear, not nearest, not
`Upsample`. The entire 28→224 (or 14→224, 7→224) reconstruction is learned
transposed convolution. No interpolation contributes to decoder latency, because
none exists.

**2. The only front-end resize is a *downsample*, and it is cheap.** PE-RGB is
average-pooled from 224×224 to the grid to be concatenated with the backbone's
feature map. It reads 150,528 elements and writes 2,352 (C2-28), performs no
multiply-accumulates, and is a memory-bound reduction — negligible against the
137 MMACs of the AE and the 0.4116 GFLOPs of the classifier. **It cannot account
for any material part of the measured latency.**

**3. C2-7 carries two bilinear interpolations that C2-14 and C2-28 do not.**
MobileViT's stage-4 block unfolds its feature map into 2×2 patches; 7 is not
divisible by 2, so timm resizes 7→8, runs attention, and resizes 8→7. This is
internal to the pretrained backbone and appears **only** at stage 4.

This is a plausible contributor to the benchmark's most counter-intuitive result:
**C2-7 has the fewest FLOPs (0.9786 corrected) yet by far the worst BS1 latency
(27.00 ms, 8.03×) — worse than C2-28 (10.20 ms, 3.03×), which has more FLOPs.**
Two extra interpolation kernels, plus a 320-channel 1×1 encoder over a 7×7 grid,
are launch-latency-bound work that FLOPs do not capture. It is offered as an
explanation consistent with the trace, **not** as a measured attribution: isolating
it would require a CUDA kernel-level profile, which has not been run.

---

## D. Effect on the spatial-information argument

**None.** The argument in `ARCHITECTURE_V2_SPATIAL_TRADEOFF.md` — that C2-28
preserves 16× more spatial detail than C2-7 because its latent is 28×28 rather
than 7×7 — is unaffected:

- the latent grids are confirmed exactly as claimed: `[64,28,28]`, `[64,14,14]`,
  `[64,7,7]`;
- no interpolation inflates any latent, so no variant is silently upsampling to
  fake resolution it does not hold;
- the reconstruction is learned in every case, so the decoder's capacity to
  recover detail is a property of the trained weights, not of a fixed resampling
  kernel.

Had the decoder ended in a bilinear resize, the spatial argument would have been
compromised — a fixed interpolation cannot recover lesion detail that the latent
does not carry. **It does not.**

---

## E. What this audit did not find

No defect that changes what any C2 variant computes. Specifically:

- classifier input is `[B, 3, 224, 224]` in all three variants — **correct**;
- the classifier is unmodified (3-channel stem) — **correct**;
- latent shapes match the design — **correct**;
- decoder stage counts match `grid → 224` by doubling — **correct**;
- the `describe()` string — **wrong**, amendment §A3;
- `tf.project` / `tf.norm` are never executed in the feature-space path —
  **dead parameters**, amendment §A4. Found during this audit, not part of the
  original brief.
