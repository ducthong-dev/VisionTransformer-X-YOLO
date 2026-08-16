# C2-28 Clean-Sanity Training Evidence

**Date:** 16 August 2026
**Status:** ⛔ **BLOCKED — the run artifacts are not present in this repository or
on this machine.** No verdict is issued, and no number below is inferred.

---

## 0. Why this document has no results in it

The 30-epoch C2-28 clean-sanity training completed on Google Colab, but its output
directory was never transferred here. Searched and confirmed empty:

| Location | Result |
|---|---|
| `results/validation/C2_28_clean_sanity` | **does not exist** |
| `results/validation/` | only local smoke runs: `A0_smoke`, `A5_smoke`, `V1_smoke_*`, `g3_smoke_test`, `override_smoke` |
| `docs/evidence/` | T4 benchmarks only |
| `~/Downloads`, `~/Desktop`, `/tmp`, repo tree | no `train_summary.json`, no `metrics.csv`, no `C2_28*` |

`$OUTPUT_ROOT` resolves to `results/` on this machine, so the Colab run's
`$OUTPUT_ROOT/validation/C2_28_clean_sanity` maps to a path that was never created
locally.

**Tasks 1, 2, 4 and 5 therefore cannot be completed**, and the adoption decision in
Task 6 cannot be made. Producing a 30-epoch table, a best-val-top-1, a convergence
verdict or an AE-behaviour conclusion without the artifacts would mean inventing
them. They are stated as unavailable instead.

## 1. What is needed — one command

From the Colab session that produced the run:

```python
!tar -czf /content/drive/MyDrive/aetfpe_v2/C2_28_clean_sanity.tar.gz \
    -C "$OUTPUT_ROOT/validation" C2_28_clean_sanity
!ls -la /content/drive/MyDrive/aetfpe_v2/
```

Then place it locally and run the ingestion:

```bash
mkdir -p results/validation
tar -xzf ~/Downloads/C2_28_clean_sanity.tar.gz -C results/validation

python scripts/ingest_training_run.py \
    --run results/validation/C2_28_clean_sanity \
    --out docs/C2_28_TRAINING_EVIDENCE.md
```

`scripts/ingest_training_run.py` produces the entire Task 1/2/4/5 analysis
automatically: the 30-epoch table (epoch, phase, train loss, train top-1/top-5,
val top-1/top-5, AE reconstruction and KL loss, lr, wall-clock, peak CUDA
allocated), best/final/convergence with a least-squares late-phase trend, the
pre-registered verdict, the efficiency summary, and the AE-behaviour section. Every
field the run did not record is printed as **ABSENT** rather than filled in.

It also **refuses to read** `eval_summary.json`, `test_*.csv/json` and
`calibration_eval_summary.json` even if present, so a directory that later gains
test evaluation cannot leak test numbers into this pre-adoption review.

If only part of the run survives, send whatever exists — `metrics.csv` and
`train_summary.json` alone are enough for Tasks 1, 2 and most of 4.

For Task 5's decoder-collapse check (needs `checkpoint.pt` and the dataset):

```bash
python scripts/inspect_ae_reconstructions.py \
    --run results/validation/C2_28_clean_sanity --n 8
```

Validation split only — the split is hard-coded to `data['val_split']` and the
script aborts if the resolved path contains `test`.

---

## 2. Evidence that IS available and does bear on C2-28

### 2.1 T4 batch-1 stability re-measurement — **[MEASURED]**

Amendment §A5 required batch-1 latency to be re-measured with lower variance
before any claim could rest on the 3× boundary. That run has completed.
Raw: `docs/evidence/2026-08-16_t4_bs1_stability_raw.json`
(sha256 `5f52f93a…de`), commit `6f71f1b`, clean tree, Tesla T4, torch 2.11.0+cu128.

| Model | mean | std | std/mean | median | **p95** | × base (mean) | × base (median) | peak MiB |
|---|---|---|---|---|---|---|---|---|
| BASELINE | 3.4648 ms | 0.7737 | 22.3 % | 3.1231 | 5.2708 | 1.000 | 1.000 | 41.61 |
| **C2-28** | **10.4558 ms** | 2.0587 | 19.7 % | 9.2627 | 14.9690 | **3.018** | 2.966 | 47.32 |

Warm-up 100, 1000 timed iterations — 2× the warm-up and 5× the samples of the
official run.

**The re-measurement confirmed the number instead of resolving the question.**

- The ratio reproduced almost exactly: **3.034 → 3.018** on the mean.
- **Variance did not fall.** Relative standard deviation went 14.5 % → 22.3 %
  (baseline) and 20.0 % → 19.7 % (C2-28). Five times the samples did not tighten
  it, so the spread is **intrinsic to the T4 at batch 1** — clock/DVFS behaviour
  and launch-overhead sensitivity on small kernels — not sampling noise that more
  iterations can average away.
- C2-28 therefore still **straddles the threshold**: 3.018× on the mean (outside
  the ≤3× preferred band), 2.966× on the median (inside it).

Per amendment §A5 the decision statistic is the **mean**, fixed in code before this
run. So `meets_preferred: false`, `hard_reject: false` — recorded by the harness
itself, which now also stamps `primary_latency_batch_size: 1` and
`primary_latency_statistic: latency_ms_mean` into the verdict.

**Consequence:** C2-28 misses the preferred latency band by **0.018×** — about
0.06 ms — against a measurement whose standard deviation is 2.06 ms. The honest
reading is that **C2-28 and the 3× threshold are indistinguishable at this
measurement's precision**. The manuscript should state the measured ratio with its
dispersion and stop there, rather than claiming either side of the boundary.

The metadata fix from commit `6f71f1b` is confirmed to have propagated: this run
records `3 x ConvTranspose2d(k=4, s=2, p=1); 28 -> 56 -> 112 -> 224`, stages 3.

### 2.2 An unexplained peak-memory discrepancy — **flagged, do not publish yet**

C2-28's batch-1 peak CUDA memory differs between the two T4 runs:

| Run | BASELINE | C2-28 |
|---|---|---|
| Official 5-model benchmark | 41.61 MiB | **65.21 MiB** |
| BS1 stability (2 models) | 41.61 MiB | **47.32 MiB** |

The baseline is **identical to the centibyte** across runs; C2-28 differs by
17.9 MiB. The likely cause is caching-allocator state — in the first run C2-28 was
built after C0's 646 MiB working set — but that is a hypothesis, not a measurement.

`peak_memory()` calls `empty_cache()` and `reset_peak_memory_stats()` before each
measurement, so this **should** have been isolated and was not.

**No C2-28 memory figure should enter the manuscript until this reproduces.**
Cheapest resolution: run the 5-model benchmark again with `--only` reordered, and
see whether C2-28's peak tracks its position in the sequence.

---

## 3. What the missing run can and cannot establish

Recorded here in advance so the conclusion cannot expand after the numbers arrive.

**Can establish:** whether the 28×28 feature-space representation is learnable for
the 39-class clean task; whether the feature-space AE trains without collapsing;
whether training is stable; measured training wall-clock and peak memory on the
training GPU.

**Cannot establish:**

- **No robustness or noise-resilience claim.** Clean data only. Reviewer #10's
  challenge to "noise-resilient latent features" is untouched by this run and needs
  Stage F.
- **No superiority over any baseline.** See §4.
- **No component attribution.** PE, transformer and AE were not ablated.
- **No test-set performance.** The test split was never constructed —
  `scripts/verify_no_test_access.py` proves it at AST level.

---

## 4. Task 3 — fair baseline comparison

> **Current fair baseline comparison is NOT YET AVAILABLE.**

The only A0 artifact in the repository is `results/validation/A0_smoke`:
**2 epochs, batch 32, macOS/MPS, 312 images, best val top-1 0.4359**, commit
`55fa0c6`, dirty tree. That is a plumbing check, not a reference.

Historical YOLOv8n-cls clean accuracy was produced under the legacy
**detection-style Protocol A** (`max_det=1, conf=0.25`, undetected images scored as
errors), a different Ultralytics version and a different environment. Per the
frozen protocol those numbers **must not** be differenced against Protocol-B
results, and **no improvement or degradation percentage against C2-28 may be
computed from them** in either direction.

Establishing this baseline is Stage A of `EXPERIMENT_CAMPAIGN_V2_PLAN.md` and is
the single highest-value next run.

---

## 5. Task 6 — adoption status

**Cannot be decided.** Criteria status:

| # | Criterion | Status |
|---|---|---|
| A | Clean classification sanity not catastrophic | **UNKNOWN — artifacts missing** |
| B | Much lighter than Original AE-TFPE | **MET [MEASURED]** — 51.0× fewer params, 34.4× fewer FLOPs, 2.79× lower batch-1 latency, 46.8× smaller, 5.80× lower batch-1 peak memory |
| C | No implementation defect remains | **MET** — A2/A4/A1/A3 applied and verified; only the cosmetic `C2` → `C2-7` JSON key rename is open |
| D | Training stable | **UNKNOWN — artifacts missing** |
| E | No test leakage | **MET** — AST-level proof, exit 0 |

Three of five are satisfied. **A and D are exactly what the missing run answers**,
so adoption turns entirely on artifacts that have not arrived.

`docs/REVISION_PROTOCOL_V2_DRAFT.md` is deliberately **not created** — the
instruction was to prepare it *if and only if* the evidence supports adoption, and
the evidence cannot currently be read.

The architecture definition that would be frozen on adoption is specified in
`MAJOR_REVISION_TWO_METHOD_STRATEGY.md` §3.5, ready to lift into the draft.
