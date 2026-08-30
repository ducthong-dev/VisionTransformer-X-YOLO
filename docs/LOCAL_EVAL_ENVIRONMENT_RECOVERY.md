# Local Evaluation Environment — Specification and Recovery

**Status as of 2026-08-30: the environment exists and is intact.** It is the conda
environment **`multimedia-reproduce`**, and it reproduces the corruption reference
byte-for-byte (26/26 configurations). Nothing needs repairing. This document exists
so that (a) the correct interpreter is always used, and (b) the environment can be
rebuilt exactly if it is ever lost.

---

## 1. Why there are two environments

| | Colab training | Local evaluation |
|---|---|---|
| Spec file | `requirements-colab.txt` | `requirements.txt` |
| Stack | Colab-native CUDA `torch`/`numpy`/`Pillow` | pinned reproducibility stack |
| Generates corruption datasets | **no** | **yes** |
| Runs final test evaluation | **no** | **yes** |
| Pixel determinism required | no | **yes** |

The split is a **packaging** boundary only. No training or evaluation protocol,
hyperparameter, seed, split, or metric definition differs between the two.

Colab must **never** install `requirements.txt`: its `numpy<2` / `pillow==10.2.0`
pins have no wheels for current Colab Pythons, so pip source-builds them and the
entire install transaction aborts, silently leaving `ultralytics` and `timm`
uninstalled. Conversely, the local machine must never evaluate under an unpinned
stack — that is what this document guards.

---

## 2. Required versions

These are not preferences. They are the environment recorded in
`docs/reproducibility_reference.json`, against which every corruption checksum was
written:

| Component | Required | Why |
|---|---|---|
| **Python** | **3.10.20** | matches the reference record |
| **NumPy** | **1.26.3** (`>=1.23,<2.0`) | corruption arithmetic; also the numpy 1.x bridge for torch 2.2 |
| **Pillow** | **10.2.0** (exact) | the `jpeg` corruption family round-trips through libjpeg; its pixel output is codec-dependent |
| libjpeg codec | `6.2` | follows from the Pillow pin |
| torch | 2.2.0 | built against numpy 1.x |
| platform | macOS arm64 | as recorded |

Only the `jpeg` family is codec-sensitive. Every other corruption is pure NumPy and
is seeded through `np.random.default_rng(PCG64)`, whose stream is stable across
NumPy versions by NumPy's own compatibility policy.

---

## 3. The environment that already exists

```bash
conda activate multimedia-reproduce
python -V          # -> Python 3.10.20
```

Its interpreter, if you need the absolute path:

```
/Users/ducthong/miniconda3/envs/multimedia-reproduce/bin/python
```

Verified 2026-08-30:

```
python 3.10.20 | numpy 1.26.3 | Pillow 10.2.0 | codec_jpeg 6.2
torch 2.2.0 | ultralytics 8.4.120 | timm 1.0.28
26/26 configurations reproduce byte-identically at the pixel level
```

---

## 4. Do not use the ambient `python3`

On this machine `python3` resolves to the **conda `base`** environment, which is
**Python 3.13.11 with NumPy 2.4.4 and Pillow 12.1.1**:

```
$ which -a python3
/Users/ducthong/miniconda3/bin/python3          <-- base, 3.13.11  DO NOT USE
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3
/usr/local/bin/python3
```

Running corruption generation or evaluation under that interpreter will silently
produce different JPEG pixels and a different environment record. It will not
crash — it will just be wrong.

**Guard rails, in order of reliability:**

1. **Always activate first.** `conda activate multimedia-reproduce` before any
   `python scripts/...` invocation in this repo.
2. **Or bypass activation entirely** — this is the safest form for scripts, cron,
   and notebooks, because it cannot be defeated by a stale shell:
   ```bash
   conda run -n multimedia-reproduce python scripts/verify_reproducibility.py --check
   ```
3. **Check before you trust a long run.** Every generation/evaluation session should
   begin with the check in §6. It takes seconds and reads the live interpreter, not
   your intent.
4. Do not `conda activate base` in this repo, and do not add the repo to a shell
   profile that auto-activates `base`.

---

## 5. Rebuilding from scratch, if the environment is ever lost

### Option A — conda (matches how the current environment was built)

```bash
conda create -n multimedia-reproduce python=3.10.20 -y
conda activate multimedia-reproduce

# torch first, from its own channel -- requirements.txt deliberately does not
# pin a torch build.
pip install torch==2.2.0 torchvision==0.17.0

pip install -r requirements.txt
```

### Option B — uv (faster; `uv` 0.7.12 is already installed)

```bash
uv venv --python 3.10 .venv-eval        # uv downloads a 3.10 toolchain if absent
source .venv-eval/bin/activate
uv pip install torch==2.2.0 torchvision==0.17.0
uv pip install -r requirements.txt
```

`.venv-eval/` is covered by the existing `.venv` ignore rule — do not commit it.

### Do not use

- **pyenv** — not installed on this machine.
- **Homebrew Python** — only 3.13 is available (`/opt/homebrew/bin/python3.13`).
- **`pip install -U`** on any of numpy / Pillow inside this environment. There is no
  version of those packages that is "safer because it is newer"; newer means
  divergent.

---

## 6. Verifying corruption reproducibility

`scripts/verify_reproducibility.py` answers the question empirically — it runs every
corruption on a fixed synthetic image (no dataset needed) and hashes the resulting
pixel arrays against `docs/reproducibility_reference.json`.

```bash
conda run -n multimedia-reproduce python scripts/verify_reproducibility.py --check
```

Expected output:

```
reference env : arm64 numpy 1.26.3 Pillow 10.2.0 jpeg 6.2
this env      : arm64 numpy 1.26.3 Pillow 10.2.0 jpeg 6.2

26/26 configurations reproduce byte-identically at the pixel level
```

**Exit codes:** `0` = every family matches. `2` = at least one family diverges or is
missing. `1` = no reference file found.

### Reading a failure

| What diverges | Meaning | Action |
|---|---|---|
| **Only the `jpeg` family** | libjpeg build differs — Pillow is not 10.2.0 | Restore the Pillow pin, or drop the jpeg family from the benchmark. **Never ship a partially-divergent benchmark.** |
| **Anything outside `jpeg`** | NumPy arithmetic differs | **Stop.** Reconcile the NumPy version before generating anything. |
| Environment line shows 3.13 / numpy 2.x | you are on `base` | See §4. |

The script distinguishes two things that are easy to conflate:

- **Pixel-array reproducibility** — the corruption arithmetic yields identical
  numbers. *This is what the benchmark requires.*
- **Encoded-file reproducibility** — the PNG/JPEG bytes on disk are identical. This
  depends on zlib and libjpeg builds, and is **neither required nor guaranteed.**

Re-writing the reference (`--write-reference`) is a deliberate, protocol-level act,
not a way to make a failing check pass. It invalidates every checksum previously
recorded against it.

---

## 7. What this environment is used for

- Corruption dataset generation (`scripts/generate_corruptions.py`)
- Checksum verification (`scripts/verify_reproducibility.py`)
- The Normal / Easy / Moderate / Hard evaluation (`scripts/evaluate.py`)
- Calibration, confusion matrices, latent analysis, complexity accounting

None of these run on Colab, and none of them are affected by the Colab dependency
change. Colab produces checkpoints; this environment consumes them.
