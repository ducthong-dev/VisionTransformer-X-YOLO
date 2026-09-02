# Authoritative Run Inventory

**Date:** 2 September 2026 (revised — A2 completed, B3 force-run)
**Source:** `AE_TFPE_MajorRevision/scientific/campaign_manifest.json` (Drive, modified
**2026-09-01T18:04:27**) and `campaign_summary.csv`, cross-checked against each run's
`run_provenance.json` and `train_summary.json`. **Transcribed, not inferred.**

Shared by every run: 50 epochs · seed 0 · batch 128 · AdamW lr 7.14e-4 · no AMP ·
`checkpoint_selection: best_val_top1` · NVIDIA A100-SXM4-40GB · git `c64a88c` ·
namespace `scientific` · `smoke_test=false` · `full_data=true` · `timing_basis=FULL_DATA` ·
dataset sha `4f8a8332…` (train 38,584 / val 8,346 / 39 classes).

## Status

| ID | Model | Total params | Trainable | Status | Runtime (s) | Best Val Top-1 | Best Val Top-5 | Verdict |
|---|---|---|---|---|---|---|---|---|
| A0 | YOLOv8n-cls RGB baseline | 1,488,247 | 1,488,247 | COMPLETED | 2282.7 | 0.9976036 | 0.9996405 | VALID |
| A1 | PE-only | 1,488,247 | 1,488,247 | COMPLETED | 2260.4 | 0.9968847 | 0.9997604 | VALID |
| A2 | TF-only (ViT-B/16) | 87,289,243 | 1,490,587 | COMPLETED | 6432.3 | 0.9964055 | 0.9996405 | **VALID** |
| A3 | PE+TF, no AE (= F3) | 87,289,243 | 1,490,587 | COMPLETED | 6423.0 | 0.9966451 | 0.9997604 | VALID |
| A4 | RGB + image-space AE | 1,747,290 | 1,747,290 | COMPLETED | 2290.2 | 0.9962856 | 0.9996405 | VALID |
| A5 | Original AE-TFPE full (= F5) | 87,549,123 | 1,750,467 | COMPLETED | 6270.1 | 0.9950875 | 0.9996405 | VALID |
| B1 | ResNet-50 baseline | 23,587,943 | 23,587,943 | COMPLETED | 3129.7 | **0.9989216** | 0.9996405 | **VALID** |
| B2 | EfficientNet-B0 baseline | 4,057,507 | 4,057,507 | COMPLETED | 2634.4 | **0.9988018** | 0.9997604 | VALID |
| B3 | ViT-B/16 baseline | 85,828,647 | 85,828,647 | COMPLETED | 14681.7 | 0.9973640 | 0.9996405 | **VALID** |
| D1 | Original AE fusion, clean objective | 87,549,123 | 1,750,467 | COMPLETED | 6741.4 | 0.9944884 | 0.9996405 | VALID |
| E3 | Efficient PE+TF, no AE | 1,637,947 | 1,488,427 | COMPLETED | 2228.4 | 0.9964055 | 0.9996405 | VALID · confounded control |
| **E5** | **Efficient AE-TFPE (C2-28)** | 1,716,586 | 1,567,066 | COMPLETED | 2305.9 | **0.9877786** | 0.9995207 | VALID |
| E7 | Efficient AE-TFPE (C2-7) | 2,544,634 | 1,593,610 | COMPLETED | 2333.9 | **0.8436377** | 0.9865804 | VALID |
| F1 | Addition fusion | 87,289,216 | 1,490,560 | COMPLETED | 6178.4 | 0.9966451 | 0.9997604 | VALID |
| F2 | Concatenation fusion | 87,289,648 | 1,490,992 | COMPLETED | 6171.0 | 0.9971244 | 0.9997604 | VALID · stem modified |
| F3 | → reuses A3 | — | — | SKIPPED_REUSE | — | — | — | LOGICAL_REUSE |
| F4 | Attention fusion | 87,289,288 | 1,490,632 | COMPLETED | 6465.7 | 0.9955667 | 0.9997604 | VALID |
| F5 | → reuses A5 | — | — | SKIPPED_REUSE | — | — | — | LOGICAL_REUSE |
| F5_clean | → reuses D1 | — | — | SKIPPED_REUSE | — | — | — | LOGICAL_REUSE |
| M1 | Legacy LUT control | 1,488,247 | 1,488,247 | COMPLETED | 2300.4 | 0.9958064 | 0.9996405 | VALID |
| M2 | Photometric (gamma) control | 1,488,247 | 1,488,247 | COMPLETED | 2284.3 | 0.9976036 | 0.9996405 | VALID |
| M3 | Augmentation control | 1,488,247 | 1,488,247 | COMPLETED | 2269.2 | 0.9968847 | 0.9997604 | VALID |

**19 VALID · 0 INCOMPLETE · 0 NOT YET RUN · 3 LOGICAL_REUSE** — 22 logical IDs,
19 physical runs, all usable. Measured GPU time ≈ 25.4 h. **The matrix is complete.**

## A2 — now COMPLETED and VALID

Re-read from Drive on 2026-09-02, from the artifacts rather than from a report of them.
Both the per-run files and the campaign manifest agree:

| Field | Value | Required | ✓ |
|---|---|---|---|
| `status` | `completed` | completed | ✓ |
| `epochs_completed` / `epochs_planned` | **50 / 50** | 50/50 | ✓ |
| `epochs_requested` | 50 | 50 | ✓ |
| `namespace` | `scientific` | scientific | ✓ |
| `smoke_test` | `false` | false | ✓ |
| `full_data` | `true` | true | ✓ |
| `limit_per_class` / train / val | all `null` | null | ✓ |
| `timing_basis` | `FULL_DATA` | FULL_DATA | ✓ |
| `dataset_sha256` | `4f8a8332…` | `4f8a8332…` | ✓ |
| `protocol_sha256` | `b06a1714…` | identical to all 16 other runs | ✓ |
| `config_sha256` | `7a7615f1…` (`configs/tf_only.yaml`) | — | recorded |
| train / val images | 38,584 / 8,346 | 38,584 / 8,346 | ✓ |
| `best_val_top1` | **0.9964054636951833** | — | recorded |
| `best_val_top5` | 0.9996405463695184 | — | recorded |
| `git_commit` | `c64a88c` | `c64a88c` | ✓ |

Manifest `end_time` 2026-09-01T18:04:14, `runtime_s` 6432.3 — consistent with the
16:17:02 start. `checkpoint.pt` (349,323,395 B) was last written 18:04:08, six seconds
*before* the completion record, which is the expected order: the final best-val
checkpoint is written, then the summary and manifest are flushed.

**Status changed INCOMPLETE → VALID.** A2 enters the evaluation on exactly the same
terms as every other physical checkpoint — same pipeline, same gate, same benchmarks.

### Three checks remain, and they are pending for every run equally

`checkpoint SHA256`, and the agreement of the in-file `epoch` / `val_top1` with the
recorded `best_val_top1`, can only be computed from the checkpoint bytes. **No run has
passed those yet, because no checkpoint has been downloaded.** They are enforced by
`scripts/verify_checkpoints.py` at acquisition, which cross-checks each artifact against
its own `train_summary.json` — for A2 that means `ckpt["val_top1"]` must equal
0.9964054636951833. Nothing is hardcoded and nothing is assumed; A2 is not special-cased.

`results/campaign/checkpoint_verification.json` does not exist yet. It is *generated* by
the verifier from real files and is deliberately **not** hand-written — fabricating an
acceptance record would defeat the gate it represents. A2 will be picked up automatically
by the directory scan.

## B1 and B3 — both completed, both VALID

Re-read from the Drive artifacts on 2026-09-02, not from a report of them. Both satisfy
every frozen metadata requirement.

| Field | B1 (ResNet-50) | B3 (ViT-B/16) | Required | ✓ |
|---|---|---|---|---|
| `status` | `completed` | `completed` | completed | ✓ |
| `epochs_completed` | **50 / 50** | **50 / 50** | 50/50 | ✓ |
| `epochs_requested` | 50 | 50 | 50 | ✓ |
| `namespace` | `scientific` | `scientific` | scientific | ✓ |
| `smoke_test` | `false` | `false` | false | ✓ |
| `full_data` | `true` | `true` | true | ✓ |
| per-class limits | all `null` | all `null` | null | ✓ |
| `timing_basis` | `FULL_DATA` | `FULL_DATA` | FULL_DATA | ✓ |
| `dataset_sha256` | `4f8a8332…` | `4f8a8332…` | `4f8a8332…` | ✓ |
| `protocol_sha256` | `b06a1714…` | `b06a1714…` | identical to all 17 others | ✓ |
| `config_sha256` | `021f3e5f…` | `c91ecb3e…` | — | recorded |
| train / val images | 38,584 / 8,346 | 38,584 / 8,346 | 38,584 / 8,346 | ✓ |
| `best_val_top1` | **0.998921639108555** | 0.9973640067098011 | — | recorded |
| runtime | 3,129.7 s | 14,681.7 s | — | recorded |
| peak CUDA | 10,888.85 MiB | 16,828.64 MiB | — | recorded |
| `git_commit` | `c64a88c` | `c64a88c` | `c64a88c` | ✓ |

B3 recovered from its epoch-2 stall and completed at 2026-09-01T22:09:14; B1 ran
immediately after, 22:09:17 → 23:01:34.

**B1 is now the highest best-validation-Top-1 arm in the entire campaign** at 0.9989216,
ahead of B2 (0.9988018). Both external baselines outrank every proposed model on clean
validation.

### The three file-level checks still run at evaluation time

`checkpoint SHA256` and the agreement of the in-file `epoch` / `val_top1` with the
recorded best require the checkpoint bytes, and are enforced by
`scripts/verify_checkpoints.py` inside the evaluation session — the same gate all nine
already-evaluated runs passed. B1 and B3 are not special-cased.

## Logical reuse

| Alias | Physical run | Basis |
|---|---|---|
| F3 (`F3_fusion_linear`) | **A3** | identical config signature |
| F5 (`F5_fusion_ae`) | **A5** | identical config signature |
| F5_clean (`F5_fusion_ae_clean`) | **D1** | identical config signature |

Inference is run **once** per physical checkpoint. Every prediction file names its
physical run; the consolidated table exposes the aliases.

## Provenance caveats

- **`git_dirty: true`** in every run's environment record. The commit is captured
  (`c64a88c`) but the working tree at training time is not exactly reconstructible.
- **Training environment** — Python 3.13.15, torch 2.11.0+cu128, Linux. **Evaluation
  environment** — conda `multimedia-reproduce`, Python 3.10.20, torch 2.2.0, macOS/MPS.
  Acceptable for inference; recorded in every evaluation output.
- **`configs/_base.yaml` still declares `epochs: 30`.** Every run used 50 via CLI
  override, and the protocol hash (`b06a1714…`) reflects 50. The file is deliberately
  **not** edited — changing it now would obscure provenance.
