# AE-TFPE — Robust Feature Fusion for Plant Leaf Disease Classification

Reproducible implementation supporting the major revision of *"Robust Feature
Fusion Model for Plant Leaf Disease Classification"* (Multimedia Tools and
Applications).

> **Read [`docs/ARCHITECTURE_RECOVERY.md`](docs/ARCHITECTURE_RECOVERY.md) first.**
> It records what the original implementation actually did, what could not be
> recovered, and which parts of the current code are reconstructions. Every claim
> in the revised manuscript should be traceable to that document or to a result
> file produced by this framework.

## What changed, and why

The code previously in this repository did not implement the method the
manuscript describes. There was no positional encoding, no transformer forward
pass, and no auto-encoder. The "feature extraction" reduced to a fixed 256-entry
pointwise lookup table blended with the original image — verified byte-exactly
against the surviving processed dataset (mean absolute error **0.0000**).

The earlier README also described a configuration that never ran. For the record:

| Old README claim | What the training log actually shows |
|---|---|
| "trained from scratch" | `pretrained=True`, "Transferred 156/158 items from pretrained weights" |
| 50 epochs | 30 epochs, 1.980 h |
| learning rate 0.01 | `optimizer=auto` resolved to AdamW at lr 7.14e-4; the log prints `ignoring 'lr0=0.01'` |
| Mosaic augmentation | present in the args but ignored by the YOLOv8 classification trainer |
| `yolov8n_cls.yaml`, `runs/train/train.log` | never existed in this repository |

This framework replaces that with a configuration-driven implementation where
every arm shares one frozen protocol.

## Layout

```
src/aetfpe/
  features/      positional_encoding.py, transformer_features.py, legacy_lut.py
  fusion/        add, concat, linear projection, attention
  autoencoder/   stacked sparse denoising AE + losses
  models/        aetfpe.py (the pipeline), classifier.py (YOLOv8n-cls & baselines)
  corruptions/   legacy Types 1-3, plus the new benchmark
  data.py  metrics.py  complexity.py  config.py  seeding.py
configs/         one YAML per arm, all inheriting _base.yaml
scripts/         train, evaluate, generate_corruptions, check_shapes,
                 print_run_matrix, analyze_complexity,
                 analyze_latent_stability, plot_latents, confusion_matrix
docs/            ARCHITECTURE_RECOVERY.md, RUNNING.md, IMPLEMENTATION_VALIDATION.md
```

## Pipeline

```
RGB [B,3,224,224] -> PE-RGB -> ViT-B/16 -> TF-RGB -> fusion -> AE -> YOLOv8n-cls -> 39 logits
```

The front-end always emits an image-space tensor in `[0,1]`, so the classifier
stays unmodified. The single exception is the plain-concatenation fusion arm,
which widens the stem to 6 channels; that is disclosed in the results table.

## Quick start

Paths come from `${DATA_ROOT}` / `${OUTPUT_ROOT}` — never hardcoded. Copy
`configs/local.example.yaml` to `configs/local.yaml` (git-ignored) or export the
variables.

```bash
export DATA_ROOT=/path/to/Plant_leaf_diseases_dataset

python scripts/check_shapes.py           # validate all 15 arms
python scripts/print_run_matrix.py       # deduplicated run plan
python scripts/generate_corruptions.py   # frozen corruption benchmark
python scripts/train.py    --config configs/aetfpe_full.yaml
python scripts/evaluate.py --run "$OUTPUT_ROOT/ablation/A5_aetfpe_full"
```

**Development runs on a laptop; training runs on Colab with CUDA.**
See [`docs/RUNNING.md`](docs/RUNNING.md) for the labelled command list for each
environment. Latency and throughput are only reportable from the CUDA
environment — results collected elsewhere are stamped
`timings_reportable: false`.

## Experiments

9 unique training runs cover the component ablation and the fusion comparison
(two fusion arms are reused rather than retrained), plus 3 mechanism controls and
3 fair baselines.

| Group | Arms |
|---|---|
| Mechanism | legacy LUT · monotonic photometric control · corruption-augmented training |
| Ablation | RGB · +PE · +TF · PE+TF (no AE) · RGB+AE · full AE-TFPE |
| Fusion | addition · concatenation · linear projection (= PE+TF) · attention · AE (= full) |
| Baselines | YOLOv8n-cls · ResNet-50 · EfficientNet-B0 · ViT-B/16 |

The mechanism controls exist because the archived YOLOv7 results show a 1.81×
robustness gain already produced by a zero-parameter lookup table. They run
first: if the full model does not beat them, no component ablation can rescue it.

## Reproducibility

Seed 0 and `deterministic=True` throughout. Corruption RNG is derived from
`(seed, relative_path, corruption, severity)`, so a corrupted file is
reproducible in isolation and identical across macOS and Linux. Every corrupted
file carries a sha256 in `corruption_manifest.csv`, and each evaluation records
the manifest's own hash, so two runs can be proven to have been scored on the
same bytes. Each run directory stores its git commit, resolved config,
environment, dataset fingerprints, metrics and checkpoint.

## Data

PlantVillage via Mendeley ([10.17632/tywbtsjrjv.1](https://doi.org/10.17632/tywbtsjrjv.1)) —
39 classes (38 disease classes plus `Background_without_leaves`),
38,584 / 8,340 / 8,335 train / val / test.

## License

MIT — see `LICENSE`.
