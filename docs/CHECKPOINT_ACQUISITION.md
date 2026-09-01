# Checkpoint Acquisition — the one step that needs you

**Status:** ⛔ **BLOCKED.** Everything downstream is built, tested and waiting.

## Why this is blocked

The campaign artifacts live in Google Drive under **`thongmxst123@gmail.com`**.
Google Drive Desktop on this MacBook is signed in as **`thong.trinh041099@gmail.com`**,
whose `My Drive` contains only `AI Developer 2022`. The files are readable through the
Drive connector (which is how the manifest, provenance and summaries were audited) but
`checkpoint.pt` is 350 MB per ViT arm and cannot come through that channel. They are not
link-shared either, so a direct fetch fails with *"Cannot retrieve the public link"*.

**No checkpoint has been fabricated, approximated, or substituted.**

## What is needed

`checkpoint.pt`, `run_provenance.json`, `train_summary.json`, `metrics.csv` for each run
below, placed at:

```
results/campaign/scientific/<RUN_ID>/
```

`last.pt` is **not** needed (it holds optimiser/scheduler state for resuming, not for
evaluation) — skipping it halves the transfer.

## Pick whichever is easiest

**Option A — share to the mounted account (recommended).** From `thongmxst123@gmail.com`,
share `AE_TFPE_MajorRevision/scientific/checkpoints` with `thong.trinh041099@gmail.com`,
then in Drive add a shortcut to *My Drive*. It syncs to
`~/Library/CloudStorage/GoogleDrive-thong.trinh041099@gmail.com/My Drive/…` and I can copy
from there directly.

**Option B — link-share.** Set the `checkpoints` folder to *Anyone with the link → Viewer*.
Then a folder fetch works unattended:

```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/1ZQmrXsL9IZtqlFI8EkyO-R2MILaZRJI4" \
      -O results/campaign/scientific
```

**Option C — from Colab, in the session that has the Drive mounted.** Smallest transfer:

```python
import os, tarfile
SRC = "/content/drive/MyDrive/AE_TFPE_MajorRevision/scientific/checkpoints"
KEEP = {"checkpoint.pt", "run_provenance.json", "train_summary.json", "metrics.csv"}
with tarfile.open("/content/drive/MyDrive/campaign_eval_bundle.tar.gz", "w:gz") as t:
    for rid in sorted(os.listdir(SRC)):
        for f in KEEP:
            p = os.path.join(SRC, rid, f)
            if os.path.exists(p):
                t.add(p, arcname=f"{rid}/{f}")
print("done")
```

then locally:

```bash
mkdir -p results/campaign/scientific
tar -xzf ~/Downloads/campaign_eval_bundle.tar.gz -C results/campaign/scientific
```

## Drive folder IDs

Parent: `1ZQmrXsL9IZtqlFI8EkyO-R2MILaZRJI4`
(`https://drive.google.com/drive/folders/1ZQmrXsL9IZtqlFI8EkyO-R2MILaZRJI4`)

| Run | Folder ID | `checkpoint.pt` size |
|---|---|---|
| A0 | `1N2Rv3gxeOKh3-UT03HwMZcvAK4WPdGYk` | ~6 MB |
| A1 | `1g2OByMZxwZf7oeQXlgAjXRF7MpBdiqTW` | ~6 MB |
| A2 | `10vpVoITvy4RfXUeMHDAZpukKYA92pbNe` | 349 MB — **incomplete, skip** |
| A3 | `1aolJrjIjOeW95vRWKK6AudHAyULzNhGw` | 350 MB |
| A4 | `1S-j1O5BbLLauEkFChwWDqrQ7qzs5IiIT` | ~7 MB |
| A5 | `15Hf5yUzfq04mpTx9oM0RtymPrl06OCCI` | 351 MB |
| B2 | `1UF5Ru06gotDv3yvLcgdwGDyejb6BvrV6` | ~16 MB |
| D1 | `1zKfHW1-58PtK-TEQZxL64ve6Fe7ERTvQ` | 351 MB |
| E3 | `1-59FubfO489aw0RRiXSHxxYPQ1L1Kb4W` | ~7 MB |
| E5 | `1p09GWDeNkQJU-yh9JT6oLjv2-F0ro96Z` | ~7 MB |
| E7 | `1w8G-APnUTTvjroj8Wp3OmcQnyNAwfNUX` | ~10 MB |
| F1 | `1iJDxc2yNHMuwgbPbh6qSQwWRUG3OqNWE` | 350 MB |
| F2 | `1YHi3eQTSTqd6wHTbgaZy4SALaJ5sc9kd` | 350 MB |
| F4 | `1qFKOaM9PS2glydxUGegmQe7nSE_6ortY` | 350 MB |
| M1 | `1LSsA0EKhHvtGzhQHeJl7n0YbKuiBdh5H` | ~6 MB |
| M2 | `1MxIU5D3MWl078qvUMlEHhE6b5OsbmyJb` | ~6 MB |
| M3 | `1Fskmtmi-C0pW4BCQ1P78eo5dJiWCG_qL` | ~6 MB |

**≈ 2.4 GB** for the 16 evaluable runs.

**Fastest useful subset (≈ 1.1 GB)** — the seven decision-critical arms, enough to answer
baseline / Original / Efficient / denoising / fusion / external baseline:
**A0, E5, B2, M1** (small, ~35 MB) plus **A5, D1, F2, F4** (~1.4 GB).

There is also `A0_protocol30_archive_1788120504`
(`1MJxTaW_Kl0e3IWRWuTdEHEbiTtjGCwfZ`) — the superseded 30-epoch A0, correctly archived
rather than overwritten. **Do not download it**; it is not part of the 50-epoch protocol.

## What happens next, automatically

Two benchmarks, reported separately and never merged.

**A — synthetic augmentation robustness** (Clean / Easy / Moderate / Hard):

```bash
conda activate multimedia-reproduce
DR=$(python -c "import yaml;print(yaml.safe_load(open('configs/local.yaml'))['env']['DATA_ROOT'])")
python scripts/verify_checkpoints.py --data-root "$DR" --strict
python scripts/evaluate_distributions.py --device mps
python scripts/paired_statistics.py
python scripts/consolidated_evidence_table.py
```

**B — Controlled Synthetic Corruption Benchmark** (6 non-geometric families x 3
severities; already frozen, spec sha `b206f1f0…`, manifest sha `cb768f09…`,
150,030 samples):

```bash
python scripts/evaluate_controlled_corruptions.py --device mps          # A0 A5 D1 E5 B2 F2 F4
python scripts/controlled_corruption_report.py
```

Add `--check-hashes` to the first command to verify every corrupted image against the
frozen pixel manifest as it is generated (slower, but proves the benchmark did not move).

### Runtime **[ESTIMATE]**

| Stage | Images / model | Small models | ViT models | Total |
|---|---|---|---|---|
| A — 4 distributions | 33,340 | ~3 min x 10 | ~15 min x 6 | **~2-3 h** |
| B — 19 distributions | 158,365 | ~11 min x 3 | ~45 min x 4 | **~3.5-4 h** |

Roughly **6-7 hours** end-to-end for both, dominated by the ViT-B/16 arms. Both runners
are resumable, so this can be stopped and restarted freely. Ordering puts the
decision-critical arms first in each: the A5-vs-D1 denoising test lands early.

`verify_checkpoints.py` refuses anything that is smoke, wrong-namespace, wrong-dataset,
not 50 epochs, or not the recorded `best_val_top1` selection. Both evaluators refuse to
run on anything not ACCEPTED, and are resumable, so a partial download is fine -- they
evaluate what has arrived and pick up the rest later.
