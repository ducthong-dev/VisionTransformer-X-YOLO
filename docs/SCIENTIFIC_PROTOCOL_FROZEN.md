# Scientific Protocol — FROZEN

**Frozen:** 15 August 2026
**Scope:** the AE-TFPE major revision for *Multimedia Tools and Applications*
**Status at freeze:** no training run, no test-set evaluation, nothing committed

This document marks the boundary between research design and execution.
Everything before it was open to revision. Everything after it is not.

---

## The five commitments

### 1. The architecture cannot change based on test results

Every component — PE-RGB, the frozen ViT-B/16 branch, the fusion operators, the
stacked sparse denoising auto-encoder, the image-space YOLO interface — is fixed
as specified in `ARCHITECTURE_RECOVERY.md` §2 and verified by
`scripts/check_shapes.py` (16/16 arms pass; every arm outputs `[0,1]`).

If a Stage-8 test result is disappointing, the response is to report it. Changing
the architecture afterwards would make every prior number a selection artifact.

The one pre-authorised exception is a **demonstrably broken** run — divergence,
NaNs, or chance-level accuracy — which may be fixed and re-run. Disappointment is
not breakage.

### 2. Corruption severities cannot change based on results

The 26 configurations in `configs/corruptions.yaml` are frozen, with exact
parameters, provenance labels and motivation recorded in `CORRUPTION_SPEC.md`.
Their sha256 is captured in every generation bundle.

No severity may be added, removed, retuned, or subset after seeing model results.
Searching for a severity at which the proposed method wins is the specific
failure this freeze exists to prevent.

Two families are implemented but deliberately excluded (`contrast`,
`motion_blur`). They stay excluded; adding them later to improve a table would be
the same failure in a different costume.

### 3. The test set cannot guide hyperparameters

| Purpose | Data |
|---|---|
| Hyperparameter selection (Stage V1) | 20% train subset → **100% validation split** |
| Robustness during development | **`corruptions_val` only** |
| Checkpoint selection | validation top-1; warm-up epochs never eligible |
| Mechanism gate G5 | `corruptions_val` |
| Final reported numbers | frozen test benchmark, **once per model, at Stage 8** |

The frozen test corruption benchmark is generated at **Stage 2B — after every
model and hyperparameter decision is locked.** It does not exist during the
selection phase, which is the cheapest possible guarantee that it cannot inform
selection.

**One prior exposure is on the record.** During implementation validation a
2-epoch smoke model was evaluated on the corrupted test split and its per-condition
top-1 was read. No parameter was changed as a result and the model was discarded,
but the observation happened and is declared in `PROVENANCE_MATRIX.md` §2.1. It
is disclosed rather than quietly dropped.

### 4. Ablation definitions cannot change post hoc

The run matrix is frozen in `PROVENANCE_MATRIX.md` §3a: **16 unique trainings**
across five groups — component ablation (6), fusion ablation (3 new + 2 reused),
denoising-objective ablation (1), mechanism controls (3), external baselines (3).

Reuse relationships are fixed: `F3 = A3`, `F5 = D1`. The fusion table's AE row is
**D1, not A5** — using A5 would confound the fusion mechanism with training-noise
exposure.

No arm may be reinterpreted, relabelled, merged, or dropped after results are
seen. An arm that produces an inconvenient result is still reported.

### 5. Failed and null results must be reported honestly

Two null results are anticipated in writing, before any data exists, so that
neither can be quietly buried:

- **A1_pe_only is expected to match A0.** The positional encoding is a fixed
  additive field identical for every image, so it carries no per-image
  information and a convolutional classifier can learn to cancel it. If A1 does
  not beat A0 beyond noise, the manuscript reports the null and reduces the PE
  claim. **PE will not be redesigned after seeing this result.**
- **A5 may fail the G5 mechanism gate.** The historical evidence shows a 1.81×
  robustness gain from a zero-parameter lookup table, against an abstract that
  claims ~1.9× for the full method. If A5 does not beat M1, M2 and M3, Stages 5–6
  do not run, and the manuscript's contribution claim is revised to describe the
  mechanism that actually works.

---

## Frozen numeric thresholds

Fixed before the corresponding results exist. **Not revisable after observing
Colab output.**

| Gate | Criterion | On failure |
|---|---|---|
| **G1** architecture | 16/16 arms pass; A0 has exactly 1,488,247 parameters; all outputs in `[0,1]`; only F2 has a 6-channel input | **STOP** |
| **G2** corruption integrity | 0 **pixel-content** mismatches; manifest sha256 recorded. Encoded-file mismatches tolerated and reported separately | **STOP** |
| **G3** baseline reproduction | val top-1 **≥ 0.990** pass · 0.980–0.990 conditional with disclosure · **< 0.980 STOP**. Also val top-5 ≥ 0.998. Reference: 0.9969 | **STOP** below 0.980 |
| **G4** mechanism sanity | M1 clean within 1 pp of A0 and M1 > A0 under pepper; A5 clean ≥ 0.95; no arm at chance | pause, fix, re-run |
| **G5** decision gate | A5 beats M1, M2 **and** M3 by **≥ 2 pp** mean corrupted top-1 on `corruptions_val` | **STOP Stages 5–6**; revise the contribution claim |

**Stage V1 candidates** — the only three permitted:
`(weight 10, warmup 3)`, `(weight 1, warmup 3)`, `(weight 10, warmup 0)`.
Selection: highest validation top-1; ties within 0.5 pp go to the simpler
configuration.

---

## Frozen terminology

Used consistently across `PROVENANCE_MATRIX.md`, `ARCHITECTURE_RECOVERY.md`,
`CORRUPTION_SPEC.md` and `EVALUATION_PROTOCOLS.md`, and to be used in the
manuscript:

| Label | Definition | Examples |
|---|---|---|
| **RECOVERED** | Historically verified implementation or protocol facts, supported by a surviving artifact | the `0.2·LUT + 0.8·x` transform (MAE 0.0000 against the surviving dataset); AdamW at lr 7.14e-4; 156/158 weight transfer; the 38,584/8,340/8,335 split; the detection-style evaluation protocol |
| **RECONSTRUCTED** | Components required to restore the intended AE-TFPE method but absent from all surviving historical code | PE-RGB; the ViT forward pass; the auto-encoder and its objective; the Type 1–3 corruption definitions |
| **NEW REVISION PROTOCOL** | New experimental controls introduced specifically to answer reviewers or to fix a methodological defect | M1/M2/M3; D1; the four new corruption families; deterministic seeded generation with pixel hashes; the single frozen training protocol; the val/test corruption separation |

---

## Frozen evidence hierarchy

**Legacy historical evidence and new Protocol-B results are never numerically
mixed.**

| | Protocol A (legacy) | Protocol B (revised) |
|---|---|---|
| Task | detection with whole-image boxes | image classification |
| Threshold | `conf=0.25`, `max_det=1`; undetected images scored as errors (42% of the archived baseline) | none; every image is scored |
| Top-5 | **undefined** | defined |
| Reproducible now? | no — the datasets are gone | yes |
| Use | citation only, explicitly labelled | every new table |

Prohibited:

- placing a Protocol-A number and a Protocol-B number in the same table without
  an explicit protocol column;
- computing any improvement percentage, ratio, or "×" factor across the two
  protocols;
- describing a Protocol-B number as an improvement over a Protocol-A number.

Protocol B is expected to give **higher** absolute accuracies simply because
confidence rejection is removed. Any such increase is a metric change, not a
method improvement, and must never be presented as one.

The historical 1.81× (9.08% → 16.40%) is **motivation** for running the mechanism
controls. It is not evidence about their outcome.

---

## Frozen reproducibility contract

| Claim | Guaranteed | Basis |
|---|---|---|
| **Pixel-array** reproducibility | **Yes**, 25/26 families | pure numpy float64; `Generator(PCG64)` is stream-stable by NumPy policy; explicit BICUBIC resampling; blur implemented in numpy rather than Pillow |
| **Encoded-file byte** reproducibility | **No** | PNG bytes depend on zlib build and encoder flags — measured: four different file hashes for identical pixels |
| `jpeg` family pixels | **environment-dependent** | libjpeg-turbo builds differ; `pillow==10.2.0` is pinned and the codec version is recorded |

`pixel_sha256` is the integrity field of record. `file_sha256` is informational.
`scripts/verify_reproducibility.py --check` compares every family against
`docs/reproducibility_reference.json` on a fixed synthetic image and must report
**26/26** before Stage 2 proceeds.

**Persistence:** corrupted pixels are ephemeral and regenerable (~21 GB, not
archived). The persistent record is the manifest, the clean split manifest, the
generation-environment bundle, the corruption config and the reproducibility
reference — tens of MB, sufficient to regenerate and verify the benchmark from
nothing.

---

## What may still change

This freeze constrains scientific decisions, not engineering. The following
remain permitted, provided they do not alter any frozen definition:

- bug fixes in code that is demonstrably wrong, with the fix and its effect recorded;
- performance work (batch size for throughput, dataloader workers, mixed precision) that does not change numerics;
- additional reporting, figures, and analyses computed from existing results;
- documentation.

Anything that changes a model, a corruption, a threshold, or an arm definition is
**not** an engineering change, regardless of how it is framed in a commit message.

---

## Sign-off

| Item | State |
|---|---|
| Architecture | frozen — `ARCHITECTURE_RECOVERY.md` §2 |
| Run matrix (16 unique) | frozen — `PROVENANCE_MATRIX.md` §3a |
| Stage V1 candidates | frozen — `PROVENANCE_MATRIX.md` §3b |
| Corruption spec (26 configs) | frozen — `CORRUPTION_SPEC.md` |
| Evaluation protocols | frozen — `EVALUATION_PROTOCOLS.md` |
| Gates G1–G5 | frozen — `COLAB_CAMPAIGN_PLAN.md` §1 |
| Hard stops | frozen — `COLAB_CAMPAIGN_PLAN.md` §0a |
| Reproducibility reference | frozen — `docs/reproducibility_reference.json` |

Next action: commit, then execute `COLAB_CAMPAIGN_PLAN.md` from Stage 0.
