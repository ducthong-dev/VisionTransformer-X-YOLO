# Preflight / Scientific Isolation, Provenance, and the A100 Readiness Gates

**Status: in force from commit that adds `src/aetfpe/provenance.py`.**

This document exists because of a near-miss. Five architecture smoke runs
(`A0 E5 M1 F1 B2`, 4 epochs, 4 images per class, 156 train images) executed
successfully on a T4 and were written to

```
AE_TFPE_MajorRevision/checkpoints/<RUN_ID>
AE_TFPE_MajorRevision/logs/<RUN_ID>.log
```

— the same directories the 30-epoch, 38,584-image scientific runs were about to
use — and the campaign manifest recorded them as `COMPLETED`. Two independent
defects followed from that:

1. the scientific campaign would have **adopted a plumbing test as a result**, and
2. the smoke epoch time (`A0: 3.2 s -> 30 epochs = 0.03 h`) was being used to
   project a campaign whose real epoch is **247x larger**.

Both are now structurally impossible rather than avoided by care.

---

## 1. Two namespaces, no shared directory

```
AE_TFPE_MajorRevision/
    preflight/                     SMOKE_TEST = True
        checkpoints/  logs/  manifest/
    scientific/                    SMOKE_TEST = False
        checkpoints/  logs/  manifest/  summaries/
    _legacy_pre_namespace/         the archived flat tree (never deleted)
```

A smoke run and the scientific run of the same arm have the **same run ID** and
write the **same file names**. Nothing about the artifact makes them
distinguishable at a glance, so the defence cannot be a naming convention — it
has to be that they are never allowed to write to the same place.

`Campaign(namespace=...)` fixes that at construction:

| | `preflight` | `scientific` |
|---|---|---|
| per-class limit | **required** | **refused** |
| `--smoke-test` | **required** | **refused** |
| artifact root | `preflight/` | `scientific/` |
| can read the other tree | **no** — the path is not on the object | **no** |

`scripts/train.py` enforces the same rule at the process boundary: it derives the
namespace from whether any `--limit-*` or `--smoke-test` was given, and refuses
an explicit `--namespace scientific` on a limited run, or `--namespace preflight`
on a full one.

Each namespace root carries a `NAMESPACE` marker file so a human reading a bare
directory on Drive can tell which half of the tree they are in.

---

## 2. Provenance: adoption and resume are checked, not assumed

Every run stamps `run_provenance.json` into its output directory, and the same
record is embedded in `last.pt` and `train_summary.json`:

| Field | Why it is identity |
|---|---|
| `run_id` | which arm this is |
| `namespace` | preflight vs scientific |
| `smoke_test` | plumbing vs science |
| `epochs_requested` | the cosine schedule's length; a different budget is a different optimiser trajectory |
| `limit_per_class`, `limit_train_per_class`, `limit_val_per_class` | which images were seen |
| `full_data` | the derived verdict, so a check cannot forget one of the three limits |
| `config_sha256` | the fully resolved config **including `--override`s**, with machine-specific keys (`data.root`, `model.num_classes`) removed so it is portable |
| `protocol_sha256` | the frozen protocol as actually applied |
| `dataset_sha256` | the train/val listing fingerprints — content identity, independent of mount point |
| `timing_basis` | `FULL_DATA` or `SMOKE_TIMING_ONLY` (see §4) |

`provenance.compare()` refuses on **any** mismatch, and on a **missing** record.
Three call sites use it:

- `Campaign.adopt_existing()` — before marking a pre-existing Drive run `COMPLETED`
- `Campaign.run()` — before copying a Drive resume point into `/content` scratch
- `train.py::_load_last()` — before restoring weights, optimiser and RNG state

A refusal prints a `REFUSED` block naming every mismatched field, and adopts,
resumes and overwrites **nothing**. It never falls back to "start from epoch 0",
because silently restarting on top of a foreign artifact is the failure it exists
to prevent.

### The rebuild

`Campaign.revalidate()` walks the manifest and demotes every `COMPLETED` run that
cannot prove it is a full-data artifact of its own namespace, recording the reason
in `manifest/revalidation_report.json`. Running it against a manifest inherited
from the flat layout is what makes the five smoke runs stop counting as completed
science.

---

## 3. Migrating the flat tree

`scripts/migrate_campaign_namespaces.py` (dry run by default, `--apply` to act).

Runs classify themselves from **their own record** — never from the run ID:

| Evidence | Verdict |
|---|---|
| `run_provenance.json` present | trust its `namespace` / `smoke_test` |
| trained images < the split it fingerprinted | `preflight` |
| full split, full epoch budget | `scientific` |
| full split, config unreadable so the epoch budget cannot be checked | `quarantine` |
| cannot prove how many images it trained on | `quarantine` |

Each run is **copied** into its namespace; the flat tree is then **moved** to
`_legacy_pre_namespace/`. Nothing is deleted.

Migrated preflight runs receive a stamp that marks them as plumbing evidence and
**deliberately omits the identity hashes**. With those fields absent, every
adoption and resume attempt refuses on "MISSING in artifact" rather than having to
reason about whether the values match.

---

## 4. Smoke timing is not full-data timing

A preflight epoch trains `4 x 39 = 156` images. A scientific epoch trains
**38,584**. A projection built on the former is not a rough estimate of the
latter; it is a different quantity in the same unit.

Under `SMOKE_TEST = True`:

- the trainer prints `SMOKE TIMING ONLY` with the subset size before the first epoch;
- the run's provenance carries `timing_basis: SMOKE_TIMING_ONLY`;
- the runner labels the measured epoch and prints **no** full-data projection;
- `Campaign.assert_projectable()` raises rather than returning a number, so the
  cost model has no path to a smoke measurement even by accident.

**The forced-tier gate (notebook Cell 15) refuses to run under `SMOKE_TEST` and
refuses to run off an A100.** It budgets six multi-hour arms from one measurement,
so that measurement is taken on full data, on the GPU that will run the tier.

---

## 5. The forced fusion set

Six **physical** trainings (`campaign.FORCED_FUSION_IDS`):

```
A5  D1  F1  F2  A3  F4
```

`A3` is included because it is the physical run behind fusion arm **F3**. `F3` has
an identical config signature (confirmed by `scripts/print_run_matrix.py` from the
signature, not from an annotation) and is satisfied by **reuse** of `A3`'s
checkpoint — it is never trained separately. Training it would train the same
model twice and report it as two experiments.

`A2`, `B1` and `B3` remain skipped: `B1`/`B3` are fully trainable and genuinely
expensive, and `B2` (EfficientNet-B0) already fills the external-baseline role.

---

## 6. Drive-only resume preflight

`scripts/preflight_resume_test.py` — runs entirely inside `preflight/`, on a
disposable `RESUME_PROBE` run, and refuses to execute outside that namespace.

1. train to epoch *N* with `--preflight-stop-after N`, mirroring to Drive
   (a deterministic stand-in for a disconnect: everything durable has just been
   written, so Drive holds exactly what a crash at that instant would leave)
2. **delete the `/content` scratch directory outright**
3. restore from **Drive and nothing else**
4. resume, and check: it announces the resume, starts at epoch *N+1*,
   `metrics.csv` is contiguous `1..E` with the pre-crash rows unchanged, the
   pre-crash best is carried in and never regresses, `checkpoint.pt` survives
5. **negative control:** offer the same directory to a different experiment
   (a different epoch budget) and check that it is **refused**

Evidence lands in `preflight/manifest/resume_test.json`.

---

## 7. The readiness gates

`src/aetfpe/preflight.py`. Each gate is written to
`preflight/manifest/preflight_gates.json` as it is proven, so the verdict survives
a runtime disconnect. The derivable gates are re-evaluated **from the artifacts on
Drive**, not from notebook variables, so a gate cannot pass because an earlier cell
was run and then edited.

```
dependency setup   CUDA   dataset counts   no-test-access
architecture smoke A / E / M / F / B
Drive persistence   Drive-only resume
preflight/scientific isolation   scientific manifest clean   A3 forced
```

All pass, and the notebook prints exactly:

```
READY FOR A100 FULL CAMPAIGN
```

otherwise exactly:

```
NOT READY FOR A100 FULL CAMPAIGN
```

followed by the failed gates and why.

---

## 8. What this did NOT change

Campaign infrastructure only. Untouched: model architectures, optimizer, LR,
weight decay, seeds, epochs, batch size, augmentation, the AE objective, the
train/val split and the test protocol. The training loop's numerics are
byte-identical; the additions are a provenance record, a namespace decision made
before any directory is created, and a refusal path where there used to be a
silent fallback.
