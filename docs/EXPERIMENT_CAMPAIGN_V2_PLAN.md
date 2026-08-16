# Experiment Campaign v2 — Staged Plan with Early-Stop Gates

**Date:** 16 August 2026 · **Status:** plan only. Nothing launched, nothing frozen.
**Supersedes:** the flat 19-run matrix in `MAJOR_REVISION_TWO_METHOD_STRATEGY.md` §7,
which was always labelled a *candidate minimum, not final*.

Evidence labels: **[MEASURED]** · **[DERIVED]** arithmetic from measurements ·
**[ESTIMATE]** reasoned projection, stated with its basis · **[NOT YET TESTED]**.

---

## 0. Design principle: buy information in the cheapest order

The 19-run matrix spends most of its GPU budget on component ablations and fusion
comparisons — work that is **worthless if the method does not beat its own
controls**. The staging below is ordered so that the cheapest run that could
falsify the contribution happens first, and every stage has an explicit condition
under which the campaign **stops** rather than continues.

The failure this protects against is concrete and already half-visible: the
recovered historical "AE-TFPE" was a zero-parameter pointwise LUT, so a plausible
outcome is that a photometric or augmentation control matches the method. If that
is true, it is far better to learn it from four runs than from nineteen.

---

## Stage A — Fair clean baseline · **1 run** · **GATE 0**

Train `A0_baseline_rgb` (YOLOv8n-cls, unmodified) under the current Protocol-B
environment, same commit, same seed, same split, same schedule as C2-28.

**Why first.** There is currently **no fair baseline**. The only A0 artifact in the
repo is a 2-epoch macOS/MPS smoke run limited to 312 images (best val top-1
0.4359) — a plumbing check, not a reference. Every historical YOLOv8n number was
produced under the legacy detection-style Protocol A, under a different
Ultralytics version, and **must not** be differenced against a Protocol-B result.
Until Stage A exists, *no* accuracy comparison for C2-28 is possible in either
direction.

This also directly answers the reviewers' fairness objection: identical protocol,
identical data, identical environment, one variable.

**Stop condition.** If A0 fails to reach a sane clean accuracy under the revised
environment, the environment — not the architecture — is the problem, and
everything downstream is uninterpretable. Fix that before spending anything else.

---

## Stage B — Mechanism gate · **4–5 runs** · **GATE 1, the decisive one**

The minimum set that can falsify the contribution.

| Run | Arm | Question it answers |
|---|---|---|
| B1 | A0 baseline *(from Stage A)* | reference |
| B2 | M1 legacy-LUT control | does the historical pointwise transform alone reproduce the effect? |
| B3 | M2 photometric control (monotonic, matched magnitude) | is any gain just a contrast/gamma shift? |
| B4 | M3 augmentation control | is any gain just extra input variability? |
| B5 | **Efficient AE-TFPE (C2-28)** *(clean-sanity run already done)* | does the method beat its own controls? |

**Original AE-TFPE is deliberately not trained at this stage** — see §Original
below.

**GATE 1.** If C2-28 does not clearly exceed the best of M1/M2/M3 on clean
validation, **stop the campaign**. Component ablations and fusion comparisons of a
method that does not beat a gamma curve are not worth GPU time, and reporting that
honestly is a stronger revision than burying it under nineteen runs.

---

## Stage C — Component ablation · **3 runs** · only if GATE 1 passes

PE contribution · transformer contribution · AE contribution, each removed
individually from Efficient AE-TFPE.

**Run these on the Efficient side, not the Original side.** RQ2/RQ3 conclusions
measured on Original AE-TFPE are **not assumed to transfer**: Efficient AE-TFPE
changes both the encoder (ViT-B/16 → MobileViT-XXS stage 2) and the AE operating
space (image → feature). Two simultaneous changes, either of which can alter which
components matter.

Note the measured prior: the PE branch's influence on the transformer features is
**stage-4-specific** (283× / 23× / 11× at stages 4/3/2 —
`ARCHITECTURE_V2_VALIDATION.md`). At stage 2, PE's effect is an order of magnitude
weaker than the configuration that finding came from, so a null PE result here
would be **expected**, not anomalous, and must be reported as such rather than
treated as a defect.

---

## Stage D — Fusion comparison · **4 runs** · only if Stage C shows the AE matters

Addition · concatenation · concat + projection · attention, against AE fusion.

**Minimum Efficient-side control needed:** at least **concat + projection** and
**attention** must be run on the Efficient architecture. Those are the two
operators that could plausibly replace the AE at the 28×28 grid, and the AE-fusion
superiority claim is the one Reviewer #12 challenges. Addition and plain
concatenation may be carried over descriptively from the Original-side matrix
**only if** Stage C shows the AE contributes at all; otherwise the whole fusion
question is moot.

---

## Stage E — Robustness benchmark · only after C/D are credible

Validation corruptions for every model decision. **Test corruptions are generated
only after all model and hyperparameter decisions are locked** — the frozen
protocol's ordering, unchanged.

---

## Stage F — Latent robustness analysis

Clean-vs-corrupted representation drift, before and after the AE. This is the
analysis that actually addresses Reviewer #10's "noise-resilient latent features"
challenge; the clean-sanity run cannot.

---

## Stage G — Complexity and final comparison

Original vs Efficient vs YOLOv8n-cls. Inference complexity is **already measured**
on the T4 and needs no re-run unless the architecture changes.

---

## Original AE-TFPE: recommendation

**Recommended: option 3 — report primarily as a computational/reference
formulation, with option 2 (controlled subset) held in reserve.**

The cost is measured, not speculative: **[MEASURED]** 87,549,123 params, 34.8731
GFLOPs, 28.51 ms T4 batch-1 latency, 344.76 ms at batch 32, 646 MiB peak. Training
it for 30 epochs costs roughly **8.5× the baseline's forward cost per image**
**[DERIVED]** from the batch-1 latency ratio.

Three reasons this is defensible rather than evasive:

1. **The efficiency argument does not need it.** Params, FLOPs, latency, throughput
   and peak memory are all hardware-independent or already measured on the
   reference GPU. Nothing about training Original AE-TFPE changes those numbers.
2. **The manuscript's own accuracy claims for it are already withdrawn** on other
   grounds — §4.6.2 does not reproduce, the blueberry results are withdrawn, Top-5
   is undefined under the recovered protocol. Retraining cannot rehabilitate
   numbers that were never reproducible.
3. **It is the honest framing.** Original AE-TFPE is the reference formulation the
   revision reconstructs; Efficient AE-TFPE is what the revision proposes.

**Escalate to option 2** (a controlled subset — same seed, same schedule, a fixed
class-stratified fraction of the training split) **if and only if** a reviewer
requires a head-to-head accuracy number. Say plainly that it is a subset and why.
**Option 1 (full training) is not recommended** unless a reviewer explicitly
demands it — it buys one accuracy number at the highest price in the campaign.

---

## Task 8 — GPU allocation: A100/L4 for training, T4 for the reference benchmark

### The distinction that makes this sound

| Metric class | Depends on hardware? | Where it must run |
|---|---|---|
| **Model quality** — accuracy, top-5, robustness, latent drift | **No.** A converged model's accuracy is a property of weights, data and schedule | any GPU; A100/L4 preferred |
| **Training wall-clock** | **Yes**, strongly | only comparable within one GPU type; never mix |
| **Inference latency / throughput / memory** | **Yes**, strongly | **T4 only**, one session, all candidates together |

**Training on A100 and reporting inference latency on T4 is scientifically
acceptable**, and is standard practice, provided three conditions hold:

1. **Determinism and schedule are held fixed** — same seed, same epochs, same batch
   size, same LR schedule, same AMP policy. Batch size especially: changing it to
   exploit A100 memory changes the optimisation trajectory and breaks comparability
   with every other arm.
2. **Every arm in a comparison is trained on the same GPU type.** Mixing an
   A100-trained C2-28 against a T4-trained A0 confounds architecture with hardware
   through numerics and non-determinism. Fix the training GPU **per comparison
   group**, not per run.
3. **Training wall-clock is never presented as architecture evidence** across
   environments. It may be reported descriptively, labelled with its hardware.

The reason this is legitimate: a trained model is a set of weights. Latency
measures how fast *those weights* execute on target hardware — a separate
measurement, correctly made on the deployment reference (T4), with the harness
already verifying on-device placement for every candidate.

**Do not force training onto T4 merely because the latency benchmark uses T4.**
That would multiply campaign cost for no scientific gain.

### Allocation

| Work | GPU | Why |
|---|---|---|
| Stage A, B, C, D training | **A100** (or L4) | quality metrics are hardware-independent; A100 shortens the critical path |
| Stage E robustness evaluation | A100 | inference over corrupted validation sets; throughput-bound |
| Stage G latency / throughput / peak memory | **T4, one session** | the frozen reference device; already measured for 5 architectures |
| Determinism checks | whichever GPU trains that group | determinism is a property of the training device |

**Caveat, and it is a real one:** at 224 px with 38,584 training images, these runs
are likely **data-loader bound rather than compute bound** — JPEG decode and
augmentation on CPU, not matrix multiply. **[ESTIMATE]** An A100 may therefore
deliver far less than its nominal speedup for the baseline and C2-28, though it
should help materially for anything carrying ViT-B/16. Measure one epoch before
committing to a GPU tier, and raise `num_workers` to match the A100 instance's CPU
count.

---

## Cost estimates

**[ESTIMATE]** — basis stated, to be replaced by the measured epoch time from the
completed C2-28 run as soon as its artifacts are available.

| Stage | Runs | Per run | Stage total |
|---|---|---|---|
| A — fair baseline | 1 | 0.5–1.5 h | **0.5–1.5 h** |
| B — mechanism gate | 3 new (M1/M2/M3) | 0.5–1.5 h | **1.5–4.5 h** |
| C — component ablation | 3 | 1–2 h | 3–6 h |
| D — fusion comparison | 2–4 | 1–2 h | 2–8 h |
| E — robustness | eval only | — | 2–4 h |
| G — T4 benchmark | 1 | ~10 min | 0.2 h |

**Through GATE 1 (Stages A + B): ≈ 2–6 GPU-hours.** That is the entire cost of
learning whether the contribution survives contact with its own controls.

Basis: 30 epochs × (38,584 train + 8,340 val) images at 224 px; baseline T4
inference throughput 4,365 img/s at batch 32 **[MEASURED]**, training taken at
roughly one third of inference throughput, then widened substantially to allow for
data-loader-bound execution. C2-28 adds ~2.5× the baseline's per-image compute
**[MEASURED]**, which lands inside this range rather than beyond it.

---

## What this plan deliberately does not do

- No test-set access at any stage before Stage E's locked decisions.
- No `corruptions_test` generation until every model and hyperparameter is locked.
- No G5, no ablation, no fusion comparison before GATE 1 passes.
- No re-freeze of `revision-protocol-v2` before the C2-28 evidence review completes.
