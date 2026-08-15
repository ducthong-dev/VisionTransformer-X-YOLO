# Colab Campaign Plan

**Frozen 15 August 2026, before any training run.** Acceptance criteria in §1 were
written before the corresponding results existed and must not be relaxed
afterwards.

Environment: Google Colab, Linux, NVIDIA CUDA. The laptop is for development
only; nothing in this document runs there.

---

## 0. Test-set discipline

**The frozen test benchmark is evaluated once per model, at Stage 8.**

| Purpose | Data |
|---|---|
| Any model or hyperparameter decision | `val` split, and `${OUTPUT_ROOT}/corruptions_val` |
| Debugging, sanity checks, calibration | same |
| Checkpoint selection | `val` top-1 |
| Final reported numbers | `test` split + `${OUTPUT_ROOT}/corruptions` — **once** |

If a Stage-8 result prompts a change to the model, the corrected model must be
re-justified on validation, and the fact that a test result influenced the
decision must be disclosed in the manuscript. There is no way to un-see a test
number; the only defence is to record that it happened.

---

## 0a. Hard stops

Two stops are unconditional. Neither may be resolved by adjusting a threshold, a
hyperparameter, or the model.

> ### STOP 1 — `G3 < 0.980`
> The baseline did not reproduce. The protocol reconstruction is wrong, and every
> arm inherits the same training loop, so no ablation number would mean anything.
> **Halt the campaign.** Diagnose the loop, the augmentation, or the schedule.
> Do not lower the threshold, do not proceed "to see what happens", and do not
> report any downstream result obtained under a failed baseline.

> ### STOP 2 — `G5` shows no meaningful robustness benefit
> A5 fails to beat M1, M2 and M3 by ≥ 2 pp mean corrupted top-1 on the validation
> corruption set. **Do not run Stages 5–6.** Do not re-tune the auto-encoder, do
> not adjust corruption severities, do not search for a severity or a subset where
> A5 wins.
>
> The correct response is editorial, not experimental: run Stage 7 for the
> baselines, then **flag the manuscript's contribution claim for revision** and
> report the mechanism that actually produces the robustness. A zero-parameter
> lookup table already delivered 1.81× historically; if that is the true
> mechanism, the paper should say so. Post-hoc tuning to rescue a claim is the
> failure mode this entire review exists to prevent.

---

## 1. Acceptance gates

### G1 — CUDA architecture validation

**Command**
```bash
python scripts/check_shapes.py
```
**Pass:** all 16 arms report `OK`; every `classifier_input` is
`[B,3,224,224]` except `F2_fusion_concat` at `[B,6,224,224]`; every
`output_range` lies within `[0,1]`; `A0_baseline_rgb` reports exactly
**1,488,247** parameters.
**Fail → STOP.** A shape or range mismatch on CUDA means the arms are not
comparable, and every downstream number would be meaningless.

### G2 — Corruption checksum validation

**Command**
```bash
python scripts/verify_reproducibility.py --check     # pixel-level, cross-platform
python scripts/generate_corruptions.py --split test
python scripts/generate_corruptions.py --split test --verify
```
**Pass:** 216,710 images verified with **0 pixel-content mismatches**; the
manifest contains 216,710 rows across 26 configurations; the manifest sha256 is
recorded. Encoded-file mismatches are tolerated and reported separately — PNG
bytes depend on the zlib build, pixel content does not.
**Fail → STOP.** A pixel mismatch means the benchmark is not reproducible, and
the fairness claim collapses.

### G3 — Baseline reproduction

**Command**
```bash
python scripts/train.py --config configs/baseline_rgb.yaml
```
**Reference:** `log-org-280223` reports val top-1 **0.9969**, top-5 0.9996 for
YOLOv8n-cls on the same split, same seed, same pretrained weights.

**Acceptance band, fixed in advance:**

| A0 val top-1 | Verdict | Action |
|---|---|---|
| **≥ 0.990** | **PASS** | Proceed |
| 0.980 – 0.990 | **CONDITIONAL** | Investigate augmentation and schedule; may proceed only if the deviation is disclosed in the manuscript's protocol section |
| **< 0.980** | **FAIL** | **STOP.** The protocol reconstruction is wrong |

Also require val top-5 ≥ 0.998.

**Why 0.7 pp.** The classifier, pretrained weights, data, split and seed are
identical — verified by matching parameter count (1,488,247) and transfer ratio
(156/158). The only differences are implementation-level: a custom loop instead
of the Ultralytics trainer, our RandAugment/erasing composition, cosine-schedule
details, and no AMP. On a task where the reference sits at 99.7%, those should
cost well under half a point. A 0.7 pp band is wide enough to absorb them and
tight enough to catch a genuine protocol error.

**Fail → STOP the whole campaign.** If the baseline does not reproduce, no
ablation result can be trusted, because every arm inherits the same loop.

### G4 — Reconstructed mechanism sanity check

Runs `M1`, `M2`, `M3`, `A5` (plus `A0` from G3), evaluated on
**`corruptions_val`**, not the test benchmark.

**Pass conditions:**

1. **M1 behaves like the historical transform.** M1 clean val top-1 within 1 pp
   of A0 — the LUT is a mild remap and should not damage clean accuracy — and
   M1 > A0 on `pepper/030` and `pepper/050`. If M1 shows no robustness advantage
   at all, the reconstruction of the historical mechanism is wrong and the
   premise of the mechanism gate needs re-examining before proceeding.
2. **A5 trains to a sensible clean accuracy**: A5 clean val top-1 ≥ 0.95. Below
   that, the AE is destroying information and Stage V1 must be revisited.
3. **No arm collapses to chance** (val top-1 > 0.10 for all four).

**Fail → PAUSE, do not stop.** Diagnose, fix, re-run the affected arm. This gate
protects against a broken reconstruction, not against an unwelcome scientific
result.

### G5 — Permission to launch the remaining runs

The decision gate. Evaluated on **`corruptions_val`**, averaged over the six
pepper severities plus the four new families at `hard`.

| Outcome | Meaning | Action |
|---|---|---|
| A5 beats **all** of M1, M2, M3 by ≥ 2 pp mean corrupted top-1 | The auto-encoder contributes beyond every trivial explanation | **Launch Stages 5–7 in full** |
| A5 beats some but not all | Partial contribution | Launch Stages 5–6; report the mechanism table prominently; soften the contribution claim |
| A5 beats **none** | The gain is explained by a lookup table, a contrast remap, or augmentation | **Do not launch Stages 5–6.** Run Stage 7 baselines only, and rewrite the paper around the mechanism that does work |

The 2 pp threshold is fixed now, before any number exists. It is chosen to exceed
plausible seed-to-seed variation on a 8,340-image validation split at these
accuracy levels.

---

## 2. Stages

Every command assumes `bash scripts/colab_setup.sh` has run and `$OUTPUT_ROOT`
is set.

**Execution order** — note that **2B runs late**, after every decision is locked:

```
0  environment
1  dataset
2A validation corruptions        <- the only corruption set that may inform a decision
3  A0 baseline reproduction      [G3]
3b V1 hyperparameter freeze
4  M1/M2/M3/A5 mechanism gate    [G4, G5]
5  component ablation
6  fusion ablation
7  external baselines
   ---- all model + hyperparameter decisions now LOCKED ----
2B frozen TEST benchmark         <- generated only now
8  final frozen-test evaluation  <- one pass per model, never repeated
9  latent-space analysis
10 complexity analysis
```

### Stage 0 — Environment verification

```bash
bash scripts/colab_setup.sh
python -c "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))"
```
**Artifacts:** `configs/local.yaml`, printed GPU name and CUDA version.
**Pass:** CUDA available; GPU name recorded (it goes into every complexity result).
**Fail → STOP.** Everything downstream depends on it.

### Stage 1 — Dataset verification

```bash
python - <<'PY'
import sys, os; sys.path.insert(0,'src')
from aetfpe.config import load_experiment
from aetfpe.data import list_classes, dataset_fingerprint
c = load_experiment('configs/_base.yaml')['data']
for s in ('train','val','test'):
    root = os.path.join(c['root'], c[f'{s}_split'])
    fp = dataset_fingerprint(root, list_classes(os.path.join(c['root'], c['train_split'])))
    print(s, fp['num_images'], fp['num_classes'], fp['listing_sha256'][:16])
PY
```
**Artifacts:** per-split fingerprints.
**Pass:** 38,584 / 8,340 / 8,335 across 39 classes. Anything else means the wrong
dataset copy is mounted — the sibling paper's copy has 8,346 / 8,334 (see
`IMPLEMENTATION_VALIDATION.md` §2.1).
**Fail → STOP.**

### Stage 2A — Validation corruptions (calibration set)

Generated **before** any training. This is the only corruption set that may
influence a decision.

```bash
python scripts/verify_reproducibility.py --check          # cross-platform pixel check
python scripts/generate_corruptions.py --split val --limit-per-class 20
python scripts/generate_corruptions.py --split val --limit-per-class 20 --verify
```
**Artifacts:** `${OUTPUT_ROOT}/corruptions_val/` (20,280 PNGs, ephemeral) plus the
persistent bundle: `corruption_manifest.csv`, `clean_split_manifest.csv`,
`generation_environment.json`.
**Pass:** `verify_reproducibility.py --check` reports **26/26** configurations
reproducing; `--verify` reports **0 pixel-content mismatches**.
**Fail → STOP.** If divergence is JPEG-only, either match the pinned
`pillow==10.2.0` or drop the jpeg family and regenerate. Do not proceed with a
partially-divergent benchmark. Divergence outside JPEG means numpy arithmetic
differs — reconcile before generating anything.

### Stage 2B — Frozen TEST benchmark

**Do not run this until Stages 3b, 4, 5, 6 and 7 are complete and every model and
hyperparameter is locked.** Generating it late is the mechanism that guarantees
it cannot be used for selection.

```bash
python scripts/generate_corruptions.py --split test          # ~21 GB, ephemeral
python scripts/generate_corruptions.py --split test --verify
```
**Artifacts:** `${OUTPUT_ROOT}/corruptions/` (216,710 PNGs, ephemeral) + the
persistent bundle.
**Pass:** **G2** — 0 pixel-content mismatches; manifest sha256 recorded in the
campaign log.
**Fail → STOP.**

> **Persistence.** Archive only the bundle (tens of MB), never the 21 GB of PNGs:
> ```bash
> tar -czf /content/drive/MyDrive/aetfpe_corruption_bundle.tar.gz \
>     -C "$OUTPUT_ROOT/corruptions" corruption_manifest.csv \
>     clean_split_manifest.csv generation_environment.json
> ```
> After a runtime reset, re-run the generator with the same seed and `--verify`
> against the archived manifest. Matching pixel hashes restore the benchmark
> exactly. The checksums, not the pixels, are the artifact that must survive.

### Stage 3 — Baseline reproduction

```bash
python scripts/train.py --config configs/baseline_rgb.yaml
```
**Artifacts:** `${OUTPUT_ROOT}/ablation/A0_baseline_rgb/{checkpoint.pt,metrics.csv,train_summary.json,config.yaml,environment.json}`.
**Pass:** **G3**.
**Fail → STOP the campaign.**

### Stage 3b — Stage V1, freeze the AE hyperparameters

```bash
# 20% training subset, FULL validation split -- the decision must be made on
# complete validation data, so only the training side is limited.
for w in 10 1; do
  python scripts/train.py --config configs/aetfpe_full.yaml --epochs 10 \
      --limit-train-per-class 200 --out "$OUTPUT_ROOT/validation/V1_w${w}_warm3" \
      --override protocol.ae_loss_weight=$w
done
python scripts/train.py --config configs/aetfpe_full.yaml --epochs 10 \
    --limit-train-per-class 200 --out "$OUTPUT_ROOT/validation/V1_w10_warm0" \
    --override protocol.ae_warmup_epochs=0
```
**Artifacts:** three run directories under `validation/`.
**Pass:** the pre-committed rule in `PROVENANCE_MATRIX.md` §2.2 selects one
setting on **validation** top-1. Write the winner into `configs/_base.yaml` and
do not revisit.
**Fail → PAUSE.** If all three sit at chance, the AE design itself is wrong.

> `--override` writes into the config **before** the protocol is built, so the
> override string is saved verbatim in the run's `config.yaml` under `_overrides`
> and in `environment.json`. A hyperparameter changed at the command line is
> therefore part of the provenance record, not an invisible flag.

### Stage 4 — Mechanism gate

```bash
for cfg in mech_legacy_lut mech_photometric mech_aug_control aetfpe_full; do
  python scripts/train.py --config configs/$cfg.yaml
done
for run in mechanism/M1_legacy_lut mechanism/M2_photometric \
           mechanism/M3_aug_control ablation/A5_aetfpe_full ablation/A0_baseline_rgb; do
  python scripts/evaluate.py --run "$OUTPUT_ROOT/$run" \
      --corruption-root "$OUTPUT_ROOT/corruptions_val"
done
```
**Artifacts:** four checkpoints; five `test_corruptions.csv` computed against
**`corruptions_val`**.
**Pass:** **G4**, then **G5**.
**Fail:** G4 → pause and fix. G5 "beats none" → do not launch Stages 5–6; go to
Stage 7, then rewrite the paper's contribution claim.

### Stage 5 — Component ablation

```bash
for cfg in pe_only tf_only pe_tf_no_ae rgb_ae; do
  python scripts/train.py --config configs/$cfg.yaml
done
```
**Artifacts:** four checkpoints under `${OUTPUT_ROOT}/ablation/`.
**Pass:** all four train to completion; none collapses to chance on validation.
**Fail → PAUSE** for the affected arm only. A single failed arm does not stop the
campaign; report it as a failed configuration if it cannot be fixed.

> Expect `A1_pe_only ≈ A0_baseline_rgb`. The positional encoding is a fixed
> additive field identical for every image, so a convolutional classifier can
> learn to cancel it (`PROVENANCE_MATRIX.md` §3.2). Pre-commit: if A1 does not
> beat A0 on validation beyond noise, report the null result. Do **not** redesign
> PE after seeing it.

### Stage 6 — Fusion ablation

```bash
for cfg in fusion_add fusion_concat fusion_attention fusion_ae_standard; do
  python scripts/train.py --config configs/$cfg.yaml
done
```
**Artifacts:** four checkpoints under `${OUTPUT_ROOT}/fusion/`.
**Pass:** all four complete; `F2_fusion_concat` logs
`classifier_stem_modified=True` (it is the one arm that changes the classifier,
and that must appear in the results table).
**Fail → PAUSE** for the affected arm.

> The fusion table uses **D1_ae_standard**, not A5, as its AE row. A5's denoising
> objective exposes it to synthetic noise that F1/F2/F4 never see; comparing A5
> against them would confound the fusion mechanism with noise exposure
> (`PROVENANCE_MATRIX.md` §3.1).

### Stage 7 — External baselines

```bash
for cfg in baseline_resnet50 baseline_efficientnet_b0 baseline_vit_b16; do
  python scripts/train.py --config configs/$cfg.yaml
done
```
**Artifacts:** three checkpoints under `${OUTPUT_ROOT}/baseline/`.
**Pass:** each reaches ≥ 0.95 val top-1 — these are strong pretrained backbones
on an easy clean task; below that indicates a training fault, not a finding.
**Fail → PAUSE** for the affected baseline. Runs regardless of the G5 outcome:
the fair-baseline table answers Reviewer #10.7 whether or not the AE survives.

### Stage 8 — Final frozen-test evaluation

**Run exactly once, after every model is final.**

```bash
for run in $(find "$OUTPUT_ROOT" -name checkpoint.pt | xargs -n1 dirname | sort); do
  python scripts/evaluate.py --run "$run" --corruption-root "$OUTPUT_ROOT/corruptions"
done
```
**Artifacts:** per run — `test_clean.json/.csv`, `test_corruptions.csv`,
`eval_summary.json` (with checkpoint sha256 and manifest sha256),
`per_class/*.json`.
**Pass:** every `eval_summary.json` carries the **same**
`corruption_manifest_sha256`, and every condition's `num_images` equals 8,335.
That is the machine-checkable proof that all models were scored on identical
bytes.
**Fail → STOP and re-run.** A manifest-hash mismatch invalidates the comparison.

### Stage 9 — Latent-space analysis

```bash
python scripts/analyze_latent_stability.py --run "$OUTPUT_ROOT/ablation/A5_aetfpe_full" \
    --corruption pepper/030 --save-embeddings
python scripts/analyze_latent_stability.py --run "$OUTPUT_ROOT/ablation/A5_aetfpe_full" \
    --corruption gaussian_noise/hard --save-embeddings
python scripts/analyze_latent_stability.py --run "$OUTPUT_ROOT/fusion/D1_ae_standard" \
    --corruption pepper/030 --save-embeddings
python scripts/plot_latents.py --run "$OUTPUT_ROOT/ablation/A5_aetfpe_full" --corruption pepper/030
```
**Artifacts:** `latent/latent_stability_*.json`, `latent/embeddings_*.npz`,
`latent/tsne_*.png`.
**Pass:** the script completes and reports `relative_drift_ratio_ae_over_pre`
together with silhouette scores for both representations.
**Fail → PAUSE.** Does not block the main tables.

> Report drift **and** separability together. A representation can trivially
> reduce drift by collapsing toward a constant, which would also destroy class
> structure. The claim "noise-resilient latent features" requires lower drift
> *with* maintained separability, and both numbers must appear.

### Stage 10 — Complexity analysis

**One session, one GPU, all arms.**

```bash
python scripts/analyze_complexity.py --device cuda --batch-size 1  --iters 100
python scripts/analyze_complexity.py --device cuda --batch-size 64 --iters 50 \
    --out "$OUTPUT_ROOT/complexity_bs64"
```
**Artifacts:** `complexity.csv`, `complexity.json` with GPU name and
`timings_reportable: true`.
**Pass:** `timings_reportable` is `true`; the same GPU name appears in both files.
**Fail → PAUSE.** If the session changes GPU mid-way, discard and re-run
everything in one session. Timings collected across GPUs are not comparable.

> Report GFLOPs **with the resolution attached**. 0.412 GFLOPs is YOLOv8n-cls at
> 224×224 counting 2×MACs; Ultralytics' own "3.4 GFLOPs" is measured at 640×640.
> Both are correct; quoting either without its resolution is not.

---

## 3. Stop-the-campaign summary

| Gate | Failure stops the campaign? |
|---|---|
| Stage 0 CUDA | **Yes** |
| Stage 1 dataset | **Yes** |
| Stage 2A reproducibility + checksums | **Yes** |
| Stage 2B / G2 checksums | **Yes** |
| Stage 3 / G3 baseline | **Yes** |
| Stage 3b V1 | No — pause and fix |
| Stage 4 / G4 sanity | No — pause and fix |
| Stage 4 / G5 decision | Not a failure. It redirects the paper |
| Stages 5–7 individual arms | No — pause that arm |
| Stage 8 manifest mismatch | **Yes**, until re-run |
| Stages 9–10 | No |

---

## 4. Budget

| Stage | Runs | T4-hours |
|---|---|---|
| 2A validation corruptions | – | ~0.2 (CPU-bound) |
| 2B test benchmark | – | ~1.5 (CPU-bound) |
| 3 baseline | 1 | 2.0 |
| 3b V1 hyperparameter freeze | 3 | ~1.0 |
| 4 mechanism gate | 4 | 9.0 |
| 5 component ablation | 4 | 9.2 |
| 6 fusion ablation | 4 | 10.0 |
| 7 external baselines | 3 | 13.0 |
| 8 final evaluation | – | ~4.0 |
| 9–10 analyses | – | ~1.5 |
| **Total, gate passes** | **19** | **≈ 51 h** |
| **Total, G5 fails** | **11** | **≈ 32 h** |

On an A100 or L4, roughly a third of those figures. Both fit inside the window to
5 September 2026, provided Stage 3 starts promptly — the schedule risk is
sequential gating, not raw compute.
