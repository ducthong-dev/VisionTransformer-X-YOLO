# Local Evaluation Handoff — MacBook Pro M4

**Date:** 23 August 2026
**Scope:** everything needed to evaluate the Colab-trained checkpoints locally against
**Normal / Easy / Moderate / Hard**.

Training happens on Colab (A100). **Evaluation happens here, on the Mac.** No
evaluation, no corruption generation and no test access occurs in
`Multimedia_Reproduce.ipynb` — verified at AST level by
`scripts/verify_no_test_access.py`.

---

## 1. What to download

One archive, written by Cell 17 of the notebook:

```
<DRIVE_ROOT>/scientific/summaries/for_local_evaluation.tar.gz
```

where `DRIVE_ROOT = /content/drive/MyDrive/AE_TFPE_MajorRevision`. It contains one
folder per **COMPLETED** run plus `campaign_summary.csv` and
`campaign_manifest.json`.

> **Only the `scientific/` namespace is ever exported.** Drive holds a second,
> physically separate tree, `preflight/`, containing smoke runs: 4 epochs on 4
> images per class, produced to prove the plumbing on a T4. Those are **not
> results**. Cell 17 refuses to run outside `scientific/` and re-checks every
> exported run's provenance stamp before adding it to the archive, so a smoke
> artifact cannot reach this machine. See `docs/PREFLIGHT_ISOLATION_PROTOCOL.md`.

Prefer this over syncing the whole Drive folder: it excludes scratch, logs and
failed runs, and is a few tens of MB rather than hundreds.

### Where each piece lives on Drive

All paths below are relative to `<DRIVE_ROOT>/scientific/`.

| Path | Contents |
|---|---|
| `manifest/campaign_manifest.json` | every run's status, params, timings, GPU, commit |
| `summaries/campaign_summary.csv` | the same as a flat table |
| `checkpoints/<ID>/checkpoint.pt` | **best val top-1** weights + resolved config + class list |
| `checkpoints/<ID>/config.yaml` | the exact resolved config, including `_overrides` |
| `checkpoints/<ID>/metrics.csv` | per-epoch loss, top-1, top-5, AE recon/KL, lr, seconds, peak CUDA |
| `checkpoints/<ID>/train_summary.json` | protocol, best val top-1, wall-clock, peak memory, environment |
| `checkpoints/<ID>/environment.json` | commit, dirty flag, library versions, dataset fingerprints |
| `checkpoints/<ID>/run_provenance.json` | run ID, namespace, smoke flag, epoch budget, per-class limits, full-data status, config/protocol/dataset hashes |
| `logs/<ID>.log` | full stdout, appended across resumes |

**The checkpoint is self-describing.** `checkpoint.pt` embeds `cfg` and `classes`, so
the evaluator rebuilds the exact architecture without you naming it — including the
Efficient arms' `tf_backbone` / `tf_stage` / `ae_space`.

## 2. Unpack

```bash
cd "/Users/ducthong/Desktop/Research 🍀/Git/VisionTransformer-X-YOLO"
mkdir -p results/campaign
tar -xzf ~/Downloads/for_local_evaluation.tar.gz -C results/campaign
ls results/campaign
```

Confirm the commit each checkpoint came from matches your checkout:

```bash
python - <<'PY'
import json, glob, subprocess
here = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
for f in sorted(glob.glob("results/campaign/*/environment.json")):
    c = json.load(open(f)).get("environment", {}).get("git_commit", "?")
    print(f"{f.split('/')[-2]:<6} {c[:12]} {'== HEAD' if c == here else '!= HEAD  <-- check'}")
PY
```

## 3. Generate the evaluation sets — **once, and only now**

The corrupted **test** sets are generated only after every model and hyperparameter
decision is locked. Training is finished, so that condition is now met.

```bash
export DATA_ROOT="/Users/ducthong/Desktop/AI/Computer Vision/Vision Transformer/dataset/Plant_leaf_diseases_dataset"
export OUTPUT_ROOT="results"

python scripts/generate_corruptions.py --config configs/corruptions.yaml --verify
```

Deterministic and checksummed: severity parameters were frozen in
`configs/corruptions.yaml` before any model was evaluated and must not be retuned
after seeing results.

⚠ **Measured: ~21 GB of disk and a long single-threaded run.** Check free space first.

### How the four evaluation levels map to the frozen benchmark

| Your level | Frozen corruption set |
|---|---|
| **Normal** | the clean `test` split — no corruption |
| **Easy** | `gaussian_noise/easy` (σ=10) · `gaussian_blur/easy` (σ=1.0) · `brightness/easy` (×0.7) · `jpeg/easy` (q=40) |
| **Moderate** | the same four families at `medium` (σ=25 · σ=2.0 · ×0.5 · q=20) |
| **Hard** | the same four families at `hard` (σ=50 · σ=4.0 · ×0.3 · q=10) |

The legacy families (`pepper`, `transparency`, `pepper_transparency` at six ratios)
reproduce the manuscript's Type 1–3 and are generated too. They are **reconstructions**
— their exact historical definitions are not recoverable — and must stay labelled as
such.

## 4. Evaluate

Per run:

```bash
python scripts/evaluate.py --run results/campaign/E5 --device mps --batch-size 64
```

All completed runs:

```bash
for d in results/campaign/*/; do
  [ -f "$d/checkpoint.pt" ] || continue
  echo "=== $(basename $d)"
  python scripts/evaluate.py --run "$d" --device mps --batch-size 64
done
```

Writes into each run directory: `eval_summary.json`, `test_clean.csv/json`,
`test_corruptions.csv`, `val_clean.csv/json`, `val_corruptions.csv`, plus `per_class/`.

**Use `--device mps`.** Apple MPS is fine for evaluation — it is a forward pass, and
inference results are hardware-independent up to floating-point ordering.

> ⚠ **Do not quote Mac timings as performance evidence.** `timing_provenance()`
> stamps every non-CUDA measurement `timings_reportable: false`. Deployment latency,
> throughput and memory come **only** from the Tesla T4 benchmarks already archived in
> `docs/evidence/`. A100 training numbers do not replace them either.

> ⚠ **MPS determinism is unresolved.** Two identical MPS runs of the baseline arm
> diverged (3.54054185 vs 3.73592884) while CPU was bit-identical. This affects
> *training*, not evaluation, but if you need bit-reproducible evaluation numbers use
> `--device cpu`.

## 5. Combined result table

```bash
python - <<'PY'
import json, glob, csv, os
rows = []
for f in sorted(glob.glob("results/campaign/*/eval_summary.json")):
    rid = f.split("/")[-2]
    e = json.load(open(f))
    tr = json.load(open(os.path.join(os.path.dirname(f), "train_summary.json")))
    rows.append({
        "id": rid,
        "model": tr.get("name"),
        "params": sum(1 for _ in []) or tr.get("model", {}).get("params_total", ""),
        "best_val_top1": tr.get("best_val_top1"),
        "clean_top1": (e.get("test_clean") or {}).get("top1"),
        "clean_top5": (e.get("test_clean") or {}).get("top5"),
    })
with open("results/campaign/combined_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(open("results/campaign/combined_results.csv").read())
PY
```

Inspect `test_corruptions.csv` inside each run for the per-severity breakdown that
fills the Easy / Moderate / Hard columns.

## 6. Protocol discipline at evaluation time

- **Protocol B only** — standard classification top-1/top-5. The historical
  detection-style Protocol A (`max_det=1, conf=0.25`) is a separate, non-comparable
  measurement. **Never difference a Protocol-A number against a Protocol-B one.**
- Report **Original AE-TFPE** and **Efficient AE-TFPE** as two named methods. Never
  merge them, and never present the Efficient variant as the originally submitted one.
- Checkpoint selection already happened on **validation**. The test split must not
  influence architecture, hyperparameters or checkpoint choice — that ordering is what
  makes the test number meaningful.
