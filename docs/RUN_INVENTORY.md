# Authoritative Run Inventory

**Date:** 2 September 2026
**Source:** `AE_TFPE_MajorRevision/scientific/campaign_manifest.json` (Drive, modified
2026-09-01T16:17:10) and `campaign_summary.csv`, cross-checked against each run's
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
| A2 | TF-only (ViT-B/16) | 87,289,243 | 1,490,587 | **RUNNING → STALLED** | — | 0.9947280 (partial) | — | **INCOMPLETE** |
| A3 | PE+TF, no AE (= F3) | 87,289,243 | 1,490,587 | COMPLETED | 6423.0 | 0.9966451 | 0.9997604 | VALID |
| A4 | RGB + image-space AE | 1,747,290 | 1,747,290 | COMPLETED | 2290.2 | 0.9962856 | 0.9996405 | VALID |
| A5 | Original AE-TFPE full (= F5) | 87,549,123 | 1,750,467 | COMPLETED | 6270.1 | 0.9950875 | 0.9996405 | VALID |
| B1 | ResNet-50 baseline | 23,587,943 | 23,587,943 | SKIPPED_SIZE | — | — | — | **NOT YET RUN** |
| B2 | EfficientNet-B0 baseline | 4,057,507 | 4,057,507 | COMPLETED | 2634.4 | **0.9988018** | 0.9997604 | VALID |
| B3 | ViT-B/16 baseline | 85,828,647 | 85,828,647 | SKIPPED_SIZE | — | — | — | **NOT YET RUN** |
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

**16 VALID · 1 INCOMPLETE · 2 NOT YET RUN · 3 LOGICAL_REUSE** — 22 logical IDs,
17 physical runs attempted, 16 usable. Measured GPU time ≈ 18.6 h.

## A2 — incomplete, excluded

`train_summary.json` last mirrored **2026-09-01T17:15:16** at
`epochs_completed: 26 / 50`, `status: running`, `train_seconds: 3340.1`,
`best_val_top1: 0.9947280`. The campaign manifest was never updated past 16:17, so the
Colab session disconnected mid-run. Nothing has been written for over 24 h.

**A2 is excluded from evaluation.** Its `checkpoint.pt` is the best of the first ≤26
epochs, not a 50-epoch result, and evaluating it would compare a half-trained model
against fully trained ones. A2 is resumable — `last.pt` carries optimiser, scheduler and
RNG state — and the evaluation runner is resumable, so A2 can be added later without
recomputing anything.

## B1 / B3 — not yet run, not frozen as missing

Both were skipped by the `MAX_TRAIN_PARAMS = 20,000,000` budget policy, **not** because
training is scientifically impossible:

- **B1 ResNet-50** — 23,587,943 params, all trainable (full backward cost).
- **B3 ViT-B/16** — 85,828,647 params, all trainable. This is the backbone the Original
  method builds on, so its absence is the more consequential of the two.

Rechecked 2026-09-02: neither has been force-run. They stay **NOT YET RUN** and will be
disclosed as missing only at the final freeze.

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
