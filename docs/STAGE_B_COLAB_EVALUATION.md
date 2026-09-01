# Stage B Evaluation on Colab — architecture, paths, setup, run order

Notebook: **`notebooks/AE_TFPE_StageB_Evaluation_Colab.ipynb`** (31 cells, 16 executable).
Runs in a **separate Colab session** from training, so evaluation and the B3 training run
proceed in parallel without touching each other.

---

## 1. Architecture

The notebook is a **thin orchestration layer**. It contains no scientific logic: every
corruption parameter, seed rule, metric, statistical test and checkpoint-selection rule
comes from the repository at a pinned commit. The notebook only decides *what to run,
in what order, and where to put the bytes*.

```
Drive (read only)                Local Colab SSD                 Drive (write)
─────────────────────            ────────────────────            ──────────────────────
scientific/checkpoints/  ──stage──▶ /content/eval_work/           evaluation/
  <RUN>/checkpoint.pt              checkpoints/<RUN>/               checkpoint_verification/
  <RUN>/run_provenance.json          (never last.pt)                benchmark_a/
  <RUN>/train_summary.json                 │                        controlled_corruptions/
  <RUN>/metrics.csv                        ▼                        predictions/
scientific/campaign_manifest.json   verify_checkpoints.py           statistics/
dataset zip ─────unzip───▶ /content/data/  │                        tables/
                                           ▼                        logs/
                                    evaluate_distributions.py       manifests/
                                    evaluate_controlled_corruptions.py
                                           │
                                   /content/eval_work/out/  ──atomic mirror──▶
```

**Why local SSD.** Inference reads 8,335 images per distribution — 4 distributions for
benchmark A and 19 for benchmark B. Doing that over the Drive FUSE mount would dominate
runtime. Checkpoints are staged once per model; the dataset is unzipped once per session.

**Why the mirror is atomic.** Each artifact is copied to `<name>.part` and then
`os.replace`d into position. A disconnect mid-copy leaves a `.part` file, which the
resume logic never mistakes for a completed result.

**Nine gates, in order.** Safe-parallel check → CUDA present → repo at pinned commit →
dependencies → dataset listing hashes → distribution structure + frozen mapping →
controlled-corruption digests → campaign manifest completion → `verify_checkpoints.py`.
Each raises `SystemExit` rather than continuing on a warning.

### Two things that would have silently corrupted results on Colab

1. **The clean↔augmented mapping is `os.listdir`-order dependent.** Rebuilding it on
   Colab's ext4 produces a *different, wrong* mapping than the macOS/APFS order the
   augmented sets were generated under, which would scramble every cross-distribution
   paired join. The proven mapping is now committed at
   `docs/evidence/clean_augmented_mapping.json` and is **authoritative**;
   `verify_eval_distributions.py` cross-checks a local rebuild against it and refuses to
   treat the rebuild as authoritative, writing it to `*_rebuilt.json` instead.
2. **JPEG corruption is libjpeg-dependent.** The notebook regenerates the controlled
   benchmark and compares 18 per-family digests against
   `docs/evidence/controlled_corruption_reference.json`. A mismatch in any **non-JPEG**
   family aborts the run as a reproducibility failure; a JPEG-only difference is recorded
   as expected. All models in a session share identical images either way, so the
   comparison stays fair.

---

## 2. Exact Drive paths

**Read only — never written, and a path guard enforces it**

| Path | Use |
|---|---|
| `/MyDrive/AE_TFPE_MajorRevision/scientific/campaign_manifest.json` | run statuses |
| `/MyDrive/AE_TFPE_MajorRevision/scientific/checkpoints/<RUN>/checkpoint.pt` | weights |
| `/MyDrive/AE_TFPE_MajorRevision/scientific/checkpoints/<RUN>/run_provenance.json` | identity |
| `/MyDrive/AE_TFPE_MajorRevision/scientific/checkpoints/<RUN>/train_summary.json` | best-val cross-check |
| `/MyDrive/AE_TFPE_MajorRevision/scientific/checkpoints/<RUN>/metrics.csv` | per-epoch history |
| `…/Plant_leaf_diseases_dataset_with_augment.zip` | dataset (3.25 GB) |

`last.pt` is **never** copied — optimiser state, useless for inference, ~1 GB for B3.

**Written — the evaluation namespace only**

| Path | Contents |
|---|---|
| `/MyDrive/AE_TFPE_MajorRevision/evaluation/checkpoint_verification/` | `checkpoint_verification.json` |
| `…/evaluation/benchmark_a/<RUN>/` | `predictions_{clean,easy,moderate,hard}.csv.gz`, aggregates, `eval_distributions.json` |
| `…/evaluation/controlled_corruptions/<RUN>/` | `predictions_<family>_<severity>.csv.gz` (19 each), aggregates, `eval_controlled.json` |
| `…/evaluation/statistics/` | `paired_statistics.json`, `controlled_report.json` |
| `…/evaluation/tables/` | `CONSOLIDATED_EVIDENCE_TABLE.md`, `CONTROLLED_CORRUPTION_RESULTS.md`, CSV |
| `…/evaluation/manifests/` | session provenance, dataset verification, distribution verification, corruption reproducibility, wave plan, session summary |
| `…/evaluation/logs/`, `…/evaluation/predictions/` | reserved |

The final cell lists any file modified under the training tree in the last two hours, so
the read-only contract is checked rather than asserted.

---

## 3. Colab runtime setup

- **Runtime → Change runtime type → GPU.** **T4 or L4 is sufficient**; A100 is not
  required and buys little, since the workload is data-loader and corruption bound.
  The notebook asserts CUDA and refuses to run on CPU.
- **Disk:** ~14 GB free on `/content` — 3.25 GB zip, ~6 GB unzipped dataset, ~2.7 GB
  checkpoints, <1 GB outputs.
- **Drive:** mounted from the account that owns `AE_TFPE_MajorRevision`.
- **Packages:** `ultralytics==8.4.120`, `timm`, `transformers==4.37.2`, `thop` are
  installed if absent. On a fresh runtime this needs **one restart**, then Run All again.
- **Runtime estimate:** benchmark A ~33 k forwards/model, benchmark B ~158 k
  forwards/model. On a T4, roughly **8–10 h** for all 17 models across both benchmarks;
  **wave 1 alone is ~2.5–3 h**. Resumable, so this can span several sessions.

**This notebook does not measure latency, throughput or memory.** Those stay with the
frozen Tesla-T4 hardware benchmark and must not be replaced by numbers from whatever GPU
Colab allocates.

---

## 4. Run All sequence

| Cell | Does | Fails if |
|---|---|---|
| 0 | config + **SAFE PARALLEL EXECUTION** check | training and evaluation roots overlap |
| 1 | GPU info, mount Drive, create `evaluation/` | checkpoint root missing |
| 2 | clone + checkout pinned commit, record provenance | CUDA unavailable |
| 3 | install deps | — (may need one restart) |
| 4 | unzip dataset, verify **6 listing hashes + dataset_sha256** | any hash or count mismatch |
| 5 | verify 4 distributions + frozen mapping | structure or mapping check fails |
| 6 | rebuild controlled benchmark, compare 18 digests | any non-JPEG family diverges |
| 7 | read manifest, build waves, append B1/B3 only if `COMPLETED 50/50` | — |
| 8 | stage checkpoints to SSD (no `last.pt`) | — |
| 9 | `verify_checkpoints.py` gate | nothing accepted |
| 10 | restore completed outputs from Drive, print dashboard | — |
| 11 | define wave runner + stats runner | — |
| 12 | **wave 1** `A0 A5 D1 E5 B2` + statistics | a run errors |
| 13 | **wave 2** `A1 A2 A3 A4` + statistics | a run errors |
| 14 | **wave 3** `F2 F4 F1 E3 E7 M1 M2 M3` (+B1/B3 if VALID) + statistics | a run errors |
| 15 | final dashboard + session manifest + read-only proof | — |

**Run All is safe on a fresh runtime and on a reconnect.** On restart it remounts Drive,
re-clones at the pinned commit, pulls completed artifacts back from Drive, and skips every
completed (model, distribution) pair. Set `FORCE_RECOMPUTE = True` only to deliberately
redo work.

## 5. Pinned commit and dataset resolution

`REPO_COMMIT = e8ca7f55d3889d2e199e23d1bdf91f36d41bd814`, verified by a fresh clone from
GitHub to contain all 27 required files (scripts, frozen protocol documents, and both
evidence artifacts). The pin necessarily points one commit back: a commit cannot contain
its own hash, so the commit that *sets* `REPO_COMMIT` is one later. That final commit
differs from the pinned one **only** in this notebook's configuration cell, and cell 2
asserts the 14 critical files exist after checkout — a stale or mistyped pin fails loudly
rather than producing quietly wrong results.

**Dataset.** Evaluation requires the three `augmented_test_images_*` sets, so it needs
`Plant_leaf_diseases_dataset_with_augment.zip` (3.25 GB) — **not** the smaller archive the
training notebook uses (`VisionTransformer_YOLO/dataset/Plant_leaf_diseases_dataset.zip`),
which carries only train/val. The augmented archive sits in **"Shared with me"** and
therefore has no `MyDrive/...` path. The notebook tries four path candidates first, in
case a shortcut exists, then falls back to fetching it by file ID
(`1zxW_UeYEYdvuRRpOWih0F5YLQUmWGbdj`) using the session's own Drive credentials.

Either way the dataset gate is the real check: six listing hashes plus `dataset_sha256`
must match the A100 training provenance, so an archive missing the augmented sets — or any
wrong copy — is rejected before a single inference runs.
