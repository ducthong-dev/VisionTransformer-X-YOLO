# Colab Execution Package — T4 Benchmark + C2-28 Clean Sanity

**Date:** 16 August 2026 · **Scope:** exactly two jobs, A and B. Nothing else.
**Prerequisite:** `MAJOR_REVISION_TWO_METHOD_STRATEGY.md` (two-method framing).

Evidence labels: **[MEASURED]** on real files/code · **[DERIVED]** arithmetic from
measurements · **[NOT YET TESTED]**.

---

## 0. What this package does and does not do

| | |
|---|---|
| **A. T4 architecture benchmark** | YOLOv8n-cls · Original AE-TFPE (C0) · C2-7 · C2-14 · C2-28 |
| **B. ONE C2-28 clean-validation sanity training** | diagnostic only |

**Explicitly NOT run:** Original AE-TFPE training · C2-7 training · C2-14 training ·
E3 · any ablation · fusion comparison · G5 · `corruptions_test` generation · any
test-set evaluation · `revision-protocol-v2` freeze.

`corruptions_val` is **not** needed here either — job B reports clean validation
only, so no corruption set is generated at all.

---

## 1. Pre-registered interpretation for job B

Fixed before execution. Do not revise after seeing the result.

**Primary question:** *Can C2-28 learn the 39-class task without catastrophic
information loss?*

This is **not** a superiority test, **not** an adoption decision, and **not** a
robustness result. C2-28 remains the *leading Efficient AE-TFPE candidate* either way.

| Clean validation top-1 | Reading | Action |
|---|---|---|
| **≥ 0.95** | The 28×28 feature-space representation is comfortably learnable | Proceed to propose the campaign. Still no superiority claim |
| **0.80 – 0.95** | Learnable but degraded relative to the ~0.99 the baseline reaches | Report the gap. Do **not** tune. Decide with the co-authors whether the Efficient variant is worth carrying |
| **< 0.80** | Catastrophic information loss | **STOP and report.** Do not tune C2-28. Use the per-epoch curve and the AE reconstruction loss to say whether the 28×28 grid or the decode-to-image step is implicated |

**Diagnostic attribution, pre-registered:** if the run fails, a *flat or rising* AE
reconstruction loss implicates the decoder; a *falling* reconstruction loss with
stagnant classification accuracy implicates the grid/interface rather than the AE.

**No post-hoc tuning under any outcome.** If C2-28 needs tuning to pass, that is
itself the result.

---

## 2. Instrumentation added for this package

Instrumentation only — no training behaviour changed. **[MEASURED]** verified
locally.

| Gap | Fix | Verified |
|---|---|---|
| Peak CUDA memory not recorded | `reset_peak_memory()` called **immediately before the first training step** (so the peak covers training, not model construction or weight download); `max_memory_allocated()` and `max_memory_reserved()` written to `train_summary.json` under `peak_memory`; running peak added per-epoch to `metrics.csv` | **[MEASURED]** returns `available: false` with the device name on CPU and MPS — no fabricated value, no substitute quantity |
| **Validation top-5 not recorded** *(second gap, found while implementing)* | `top5` computed in `run_epoch` for train and val | **[MEASURED]** `train_top5` / `val_top5` now present in `metrics.csv` |

Top-5 is computed under `torch.no_grad()` on **detached** logits, so it cannot
reach the graph, the gradients or the optimiser state.

MPS is deliberately reported as *unavailable* rather than approximated: `torch.mps`
exposes no counter with the same semantics as CUDA's allocated/reserved pair.

---

## 3. A determinism problem you should know about before running

**[MEASURED]** while testing the instrumentation, using the **baseline arm only**
(no AE, no timm encoder — pure frozen-v1 code paths, so this predates every change
made for the v2 investigation):

| Device | Two runs, identical seed and config | Result |
|---|---|---|
| **CPU** | `train_loss` 3.73670347 vs 3.73670347 | **identical** ✓ |
| **MPS** | `train_loss` 3.54054185 vs 3.73592884 | **divergent** ✗ |
| **CUDA** | — | **[NOT YET TESTED]** |

The frozen protocol asserts `seed=0, deterministic=True`. That holds on CPU. It
does **not** hold on Apple MPS. **Whether it holds on CUDA is unknown**, and it
matters: the reproducibility claim in `SCIENTIFIC_PROTOCOL_FROZEN.md` depends on it.

This is not caused by the instrumentation — the divergent arm uses no new code —
and no scientific conclusion currently rests on an MPS number (all MPS runs to date
were plumbing checks, labelled as such).

**Cell 3 below tests CUDA determinism in ~2 minutes, before anything expensive
runs.** `seed_everything()` sets `cudnn.deterministic=True` but not
`torch.use_deterministic_algorithms(True)`; strengthening that would change
training behaviour, so it is **not** done here — it is a decision to make after the
check, with evidence.

---

## 4. Copy-paste Colab cells, in execution order

### Cell 1 — repository

```python
!git clone https://github.com/ducthong-dev/VisionTransformer-X-YOLO.git
%cd VisionTransformer-X-YOLO
!git log -1 --format="%H  %s"
```

### Cell 2 — environment + dataset

```python
from google.colab import drive; drive.mount('/content/drive')

import os
os.environ["DATA_ROOT"]   = "/content/data/Plant_leaf_diseases_dataset"
os.environ["OUTPUT_ROOT"] = "/content/output"

!mkdir -p /content/data /content/output
# adjust the archive path to wherever the dataset lives in your Drive
!unzip -q "/content/drive/MyDrive/VisionTransformer_YOLO/dataset/Plant_leaf_diseases_dataset.zip" -d /content/data

!bash scripts/colab_setup.sh
!python scripts/stage0_environment.py --require-cuda
!python scripts/verify_dataset.py
```

**STOP if:** `stage0_pass: false` (dirty repo, no CUDA, or Pillow ≠ 10.2.0), or
`verify_dataset.py` exits non-zero. Expected split: **38,584 / 8,340 / 8,335**
across 39 classes. A different split means the wrong dataset copy is mounted.

### Cell 3 — CUDA determinism check (~2 min) · **new, run before anything expensive**

```python
!for i in 1 2; do \
    python scripts/train.py --config configs/baseline_rgb.yaml --device cuda \
      --epochs 2 --limit-train-per-class 2 --limit-val-per-class 2 \
      --batch-size 8 --num-workers 0 \
      --out "$OUTPUT_ROOT/validation/cudadet$i" >/dev/null 2>&1; \
  done

import csv
a = list(csv.DictReader(open('/content/output/validation/cudadet1/metrics.csv')))
b = list(csv.DictReader(open('/content/output/validation/cudadet2/metrics.csv')))
keys = [k for k in a[0] if k not in ('seconds', 'peak_cuda_alloc_mb')]
identical = all(a[i][k] == b[i][k] for i in range(len(a)) for k in keys)
print("CUDA deterministic across identical runs:", identical)
for i in range(len(a)):
    print(f"  ep{a[i]['epoch']}: {a[i]['train_loss'][:12]} vs {b[i]['train_loss'][:12]}")
```

**If `False`:** do not stop the package — jobs A and B are still informative — but
**record it**, and treat it as a blocking issue for `revision-protocol-v2`, since
the frozen reproducibility claim would not hold on the training device.

### Cell 4 — JOB A: T4 architecture benchmark

```python
!python scripts/benchmark_architectures.py --device cuda --pretrained \
    --only BASELINE C0 C2 C2-14 C2-28 \
    --batch-sizes 1 32 --warmup 50 --iters 200 \
    --out "$OUTPUT_ROOT/architecture_v2/t4_benchmark.json"
```

Runs all five models in **one process on one GPU**, so the comparison is internally
consistent. Identical AMP policy across candidates (off by default, matching the
training protocol).

```python
# readable summary
import json
d = json.load(open('/content/output/architecture_v2/t4_benchmark.json'))
print("GPU:", d['hardware'].get('gpu'), "| timings reportable:", d['timings_reportable'])
hdr = f"{'candidate':<10}{'params':>12}{'GFLOPs':>9}{'bs1 ms':>9}{'bs32 ms':>10}{'img/s':>10}{'peak MiB':>10}"
print(hdr); print('-'*len(hdr))
for r in d['candidates']:
    l1, l32 = r['latency']['1'], r['latency']['32']
    print(f"{r['candidate']:<10}{r['params_total']:>12,}{r['gflops']:>9.4f}"
          f"{l1['latency_ms_mean']:>9.3f}{l32['latency_ms_mean']:>10.3f}"
          f"{l32['throughput_img_per_s']:>10.1f}{l32.get('peak_gpu_mem_mb', float('nan')):>10.1f}")
```

### Cell 5 — JOB B: ONE C2-28 clean-validation sanity training

```python
!python scripts/train.py --config configs/aetfpe_full.yaml \
    --override model.tf_backbone=mobilevit_xxs \
    --override model.ae_space=feature \
    --override model.tf_stage=2 \
    --out "$OUTPUT_ROOT/validation/C2_28_clean_sanity"
```

Uses the frozen split, seed, image size and training policy. Touches **no** test
data and **no** corruption set — `train.py` contains no reference to the test split.

```python
# pre-registered readout
import json, csv
s = json.load(open('/content/output/validation/C2_28_clean_sanity/train_summary.json'))
rows = list(csv.DictReader(open('/content/output/validation/C2_28_clean_sanity/metrics.csv')))

print(f"best clean val top-1 : {s['best_val_top1']:.4f}")
print(f"total wall-clock     : {s['train_seconds']/60:.1f} min")
print(f"peak CUDA memory     : {s['peak_memory']}")
print()
print(f"{'ep':>3}{'stage':>11}{'tr_loss':>10}{'tr_top1':>9}{'val_top1':>10}{'val_top5':>10}{'ae_recon':>10}{'sec':>8}")
for r in rows:
    print(f"{r['epoch']:>3}{r['stage']:>11}{float(r['train_loss']):>10.4f}"
          f"{float(r['train_top1']):>9.4f}{float(r['val_top1']):>10.4f}"
          f"{float(r['val_top5']):>10.4f}"
          f"{(float(r['train_ae_recon']) if r.get('train_ae_recon') else float('nan')):>10.4f}"
          f"{float(r['seconds']):>8.1f}")

t1 = s['best_val_top1']
print("\nPRE-REGISTERED VERDICT:",
      "PASS -- learnable"           if t1 >= 0.95 else
      "DEGRADED -- report, do not tune" if t1 >= 0.80 else
      "CATASTROPHIC -- STOP and report")
```

### Cell 6 — persist to Drive

```python
!mkdir -p /content/drive/MyDrive/aetfpe_v2
!cp "$OUTPUT_ROOT/architecture_v2/t4_benchmark.json" /content/drive/MyDrive/aetfpe_v2/
!tar -czf /content/drive/MyDrive/aetfpe_v2/C2_28_clean_sanity.tar.gz \
    -C "$OUTPUT_ROOT/validation" C2_28_clean_sanity
!ls -la /content/drive/MyDrive/aetfpe_v2/
```

---

## 5. Expected output files

| Path | Contents | Size |
|---|---|---|
| `$OUTPUT_ROOT/environment/stage0_environment.json` | git commit + dirty flag, GPU, CUDA/cuDNN, torch, ultralytics, numpy, Pillow, JPEG/zlib | ~2 KB |
| `$OUTPUT_ROOT/dataset/dataset_manifest.csv` | one row per image, 55,259 rows | **[MEASURED]** 5.2 MB |
| `$OUTPUT_ROOT/dataset/dataset_summary.json` | counts, per-class breakdown, manifest sha256 | ~8 KB |
| `$OUTPUT_ROOT/architecture_v2/t4_benchmark.json` | per-candidate params, FLOPs, latency b=1/32, throughput, peak GPU memory, shapes, interfaces | ~60 KB |
| `$OUTPUT_ROOT/validation/C2_28_clean_sanity/checkpoint.pt` | best-val-top1 weights + cfg + class list | **[MEASURED]** **7.27 MB** |
| `…/metrics.csv` | per-epoch: loss, top-1, **top-5**, AE recon, AE KL, lr, stage, seconds, peak CUDA MiB | ~4 KB |
| `…/train_summary.json` | protocol, best val top-1, wall-clock, **peak_memory**, environment | ~8 KB |
| `…/config.yaml`, `…/environment.json` | resolved config incl. `_overrides`; provenance | ~10 KB |
| `$OUTPUT_ROOT/validation/cudadet{1,2}/` | determinism check artifacts | ~12 MB |

### Checkpoint sizes · **[MEASURED]** by saving real `train.py` payloads

| Model | Params | Checkpoint |
|---|---|---|
| YOLOv8n-cls baseline | 1,488,247 | **5.76 MB** |
| Original AE-TFPE (C0) | 87,549,123 | **334.72 MB** |
| **C2-28** | 1,716,739 | **7.27 MB** |

Job B produces **one** checkpoint: 7.27 MB. C0's 334.72 MB is listed for
comparison only — Original AE-TFPE is **not trained** in this package.

---

## 6. Disk requirement · **[DERIVED]** from measured sizes only

| Item | Size | Source |
|---|---|---|
| Dataset, unzipped | **927 MB** | **[MEASURED]** `du -sh` on the real dataset |
| ViT-B/16 in21k weights (HF cache) — job A only | **342 MB** | **[MEASURED]** local HF cache |
| MobileViT-XXS weights (timm/HF cache) | **4.9 MB** | **[MEASURED]** local HF cache |
| `yolov8n-cls.pt` | **5.3 MB** | **[MEASURED]** local file |
| Job B checkpoint | **7.3 MB** | **[MEASURED]** |
| Determinism-check checkpoints (2×5.76 MB) | **11.5 MB** | **[DERIVED]** |
| Dataset manifest + all JSON/CSV outputs | **~5.3 MB** | **[MEASURED]** |
| **Total working set** | **≈ 1.31 GB** | **[DERIVED]** |
| Drive archive (benchmark JSON + sanity run) | **≈ 8 MB** | **[DERIVED]** |

Plus the zipped dataset if it is copied locally before unzipping. Comfortably
within a standard Colab disk allocation. **No corruption sets are generated**, so
the ~21 GB test benchmark is not part of this package.

---

## 7. STOP conditions

| # | Condition | Action |
|---|---|---|
| 1 | `stage0_environment.py` reports `stage0_pass: false` | **STOP.** Dirty repo, missing CUDA, or wrong Pillow — later artifacts could not be tied to a commit |
| 2 | `verify_dataset.py` exits non-zero | **STOP.** Wrong dataset copy; the sibling ResCBAM copy has 8,346 / 8,334 |
| 3 | Job A: GPU name differs between the two batch sizes, or the session changes GPU mid-run | **STOP and re-run job A in one session.** Cross-GPU timings are not comparable |
| 4 | Job A: C2-28 latency **> 5×** baseline | **Do not stop the package.** Record it; the Efficient-variant efficiency claim needs reconsideration and Option B moves back into scope |
| 5 | Job B: clean val top-1 **< 0.80** | **STOP after job B.** Report, apply the §1 diagnostic attribution, **do not tune** |
| 6 | Job B: loss becomes NaN or diverges | **STOP.** Report as a training defect, not an architecture result |
| 7 | Cell 3: CUDA determinism `False` | Continue, but record as a blocking issue for `revision-protocol-v2` |
| 8 | Any temptation to run an ablation, G5, a corruption set, or a test evaluation | **STOP.** Out of scope for this package |

---

## 8. What must not be estimated from the Mac

**[NOT YET TESTED]** and deliberately absent from every document until job A runs:
T4 latency at batch 1 · T4 latency at batch 32 · throughput · peak CUDA
allocated/reserved memory — for **all five** models including the baseline.

Parameters and GFLOPs are hardware-independent and already final; they are the
only complexity figures currently quotable.

---

## 9. After the package

Return the job A summary table and the job B readout. Then, and only then:

1. Apply the §1 pre-registered verdict to C2-28.
2. Decide whether an Efficient-side fusion control is needed — the 19-run matrix
   remains a **candidate minimum, not final**, because RQ2/RQ3 conclusions measured
   on Original AE-TFPE are **not assumed to transfer** to Efficient AE-TFPE, which
   changes both the encoder and the AE operating space.
3. Resolve the determinism finding before any `revision-protocol-v2` freeze.

No re-freeze, no campaign, no further architecture redesign.
