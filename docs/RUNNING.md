# Running the AE-TFPE framework

Two environments, one codebase. Every command below is labelled with where it
belongs. **Nothing expensive runs on the laptop.**

| | LOCAL MACBOOK | GOOGLE COLAB |
|---|---|---|
| OS | macOS, Apple Silicon | Linux |
| Device | CPU / MPS | NVIDIA CUDA (T4, L4, A100) |
| Role | audit, implement, unit test, smoke test | **source of truth** for training and all runtime measurements |
| Allowed | shape checks, tiny corruption samples, 1–2 epoch smoke tests on a few images per class | full corruption generation, full training, full evaluation, latency/throughput |

> **Dependency specs are not interchangeable.** Local uses `requirements.txt`
> (pinned: Python 3.10.x / NumPy 1.26.x / Pillow 10.2.0, for corruption-pixel
> reproducibility). Colab uses `requirements-colab.txt` (training deps only, on
> Colab's native CUDA stack) — `scripts/colab_setup.sh` handles this. Installing
> `requirements.txt` on Colab aborts the whole install. See
> [LOCAL_EVAL_ENVIRONMENT_RECOVERY.md](LOCAL_EVAL_ENVIRONMENT_RECOVERY.md).

## Paths

Committed configs contain **no absolute paths**. They reference `${DATA_ROOT}`,
`${OUTPUT_ROOT}`, `${CACHE_ROOT}`, `${MODEL_ROOT}`, resolved from (in order)
environment variables, then a git-ignored `configs/local.yaml`, then defaults.

```yaml
# configs/local.yaml  (git-ignored; copy from configs/local.example.yaml)
env:
  DATA_ROOT: "/absolute/path/to/Plant_leaf_diseases_dataset"
  OUTPUT_ROOT: "results"
```

The identical config file therefore runs in both environments with only the
roots changing.

---

## LOCAL MACBOOK — development and validation

```bash
# one-time
export DATA_ROOT="$HOME/path/to/dataset/Plant_leaf_diseases_dataset"
# or write configs/local.yaml

# 1. tensor-shape validation for all 15 arms   (seconds)
python scripts/check_shapes.py

# 2. deduplicated run matrix                   (instant)
python scripts/print_run_matrix.py

# 3. tiny corruption sample + determinism check (~30 s, ~130 MB)
python scripts/generate_corruptions.py --limit-per-class 2
python scripts/generate_corruptions.py --limit-per-class 2 --verify

# 4. smoke-test training, 2 epochs, 8 images/class  (~1 min baseline)
python scripts/train.py --config configs/baseline_rgb.yaml \
    --epochs 2 --limit-per-class 8 --batch-size 32 --num-workers 2 \
    --out results/validation/A0_smoke

# 5. smoke-test the full model, 8 epochs (3 AE warm-up + 5 joint)  (~4 min)
python scripts/train.py --config configs/aetfpe_full.yaml \
    --epochs 8 --limit-per-class 8 --batch-size 32 --num-workers 2 \
    --out results/validation/A5_smoke

# 6. smoke-test evaluation
python scripts/evaluate.py --run results/validation/A0_smoke \
    --limit-per-class 4 --batch-size 32 --num-workers 2

# 7. hardware-INDEPENDENT complexity only (params + FLOPs)
python scripts/analyze_complexity.py --skip-latency
```

**Never run locally:** the full corruption benchmark, any full-dataset training,
the ablation matrix, or latency benchmarking. `analyze_complexity.py` without
`--skip-latency` will still run on MPS, but it stamps every result
`timings_reportable: false` and prints a warning — those numbers are a sanity
check on the implementation, never manuscript evidence.

---

## GOOGLE COLAB — full experiments

```python
# cell 1 — repo + dataset
!git clone https://github.com/ducthong-dev/VisionTransformer-X-YOLO.git
%cd VisionTransformer-X-YOLO
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/data
!unzip -q "/content/drive/MyDrive/VisionTransformer_YOLO/dataset/Plant_leaf_diseases_dataset.zip" -d /content/data
```

```bash
# cell 2 — environment
!bash scripts/colab_setup.sh
```

```bash
# cell 3 — freeze the corruption benchmark ONCE, then reuse for every model
!python scripts/generate_corruptions.py
!python scripts/generate_corruptions.py --verify        # 0 mismatches expected
# archive it so a runtime reset does not change the benchmark:
!tar -czf /content/drive/MyDrive/aetfpe_corruptions.tar.gz -C /content/output corruptions
```

Disk: the full benchmark is **26 configurations × 8,335 images = 216,710 PNGs
≈ 21 GB** (measured at 94.5 KB/file from the smoke sample). Colab's `/content`
has room; a laptop does not. Generate the val calibration set too (another
21 GB) only if you need it at full size — `--limit-per-class 20` is enough for
calibration. If space is tight, generate the legacy families first with
`--only pepper transparency pepper_transparency`.

```bash
# cell 4 — mechanism controls FIRST (the decision gate, see ARCHITECTURE_RECOVERY.md §4.1)
for cfg in mech_legacy_lut mech_photometric mech_aug_control baseline_rgb aetfpe_full; do
  python scripts/train.py --config configs/$cfg.yaml
  python scripts/evaluate.py --run "$OUTPUT_ROOT/$(python -c "
import sys,os;sys.path.insert(0,'src')
from aetfpe.config import load_experiment
c=load_experiment('configs/$cfg.yaml');print(os.path.join(c['group'],c['name']))")"
done
```

```bash
# cell 5 — remaining ablation + fusion arms, only if the gate passes
for cfg in pe_only tf_only pe_tf_no_ae rgb_ae fusion_add fusion_concat fusion_attention; do
  python scripts/train.py --config configs/$cfg.yaml
done

# cell 6 — fair baselines (run regardless of the gate)
for cfg in baseline_resnet50 baseline_efficientnet_b0 baseline_vit_b16; do
  python scripts/train.py --config configs/$cfg.yaml
done
```

```bash
# cell 7 — analyses, on CUDA, in one session so timings are comparable
!python scripts/analyze_complexity.py --device cuda --batch-size 1
!python scripts/analyze_complexity.py --device cuda --batch-size 64   # throughput
!python scripts/analyze_latent_stability.py --run $OUTPUT_ROOT/ablation/A5_aetfpe_full \
    --corruption pepper/030 --save-embeddings
!python scripts/plot_latents.py --run $OUTPUT_ROOT/ablation/A5_aetfpe_full --corruption pepper/030
!python scripts/confusion_matrix.py --run $OUTPUT_ROOT/ablation/A0_baseline_rgb --condition clean
!python scripts/confusion_matrix.py --run $OUTPUT_ROOT/ablation/A5_aetfpe_full --condition pepper_050
```

```bash
# cell 8 — persist everything before the runtime dies
!tar -czf /content/drive/MyDrive/aetfpe_results.tar.gz -C /content output
```

**Latency must be measured in a single Colab session**, on one GPU, for every
arm, at one batch size and resolution. Colab hands out T4 / L4 / A100
unpredictably, so timings collected across sessions are not comparable. The GPU
name is recorded in `complexity.json`.

---

## Reproducibility contract

- Seed 0, `deterministic=True`, everywhere.
- Corruption RNG is derived from `(seed, relative_path, corruption, severity)` via
  blake2b — not from iteration order or worker count — so a corrupted file is
  reproducible in isolation and identical on macOS and Linux.
- `corruption_manifest.csv` carries a sha256 per file; `evaluate.py` stamps the
  manifest's own sha256 into every result, so two runs can be *proven* to have
  been scored on the same bytes.
- Every run directory records the git commit, dirty flag, torch version,
  platform, device, resolved protocol, and per-split dataset fingerprints.
- No manuscript number should be typed by hand; all are derivable from the CSVs.
