# Protocol Amendment Record — 16 August 2026

**Status:** PROPOSED. Nothing here is applied. No code, no rule, no JSON and no
frozen protocol has been changed by this document.

**Trigger:** the official Tesla T4 benchmark (first CUDA run of
`scripts/benchmark_architectures.py`) plus two commissioned audits — decision-rule
consistency, and the decoder shape path.

**Why this file exists.** Three of the four items below were discovered *after*
observing the T4 results. Any change to a decision rule, a measurement convention
or a reported figure made after seeing results must be recorded as an explicit,
dated amendment with its justification and its effect on every candidate —
otherwise it is indistinguishable from fitting the rule to the answer. Each item
therefore states what it changes, what it does **not** change, and whether it
moves any candidate across any threshold.

Evidence labels: **[MEASURED]** on real code/hardware · **[DERIVED]** arithmetic
from measurements · **[AWAITING DATA]**.

---

## Summary of the four items

| # | Item | Class | Changes a verdict? | Applied? |
|---|---|---|---|---|
| **A1** | Latency rule does not name a batch size | Protocol ambiguity | **Yes** — see §A1.5 | No |
| **A2** | `thop` over-counts `ConvTranspose2d` by stride² | Measurement defect | **No** | No |
| **A3** | `SlimFeatureSpaceAE.describe()` reports the wrong decoder depth | Documentation defect | No | No |
| **A4** | `tf.project` / `tf.norm` are dead parameters in every C2-* candidate | Implementation defect | No | No |
| **A5** | Latency rule does not say mean or median, and BS1 std is 15–30 % | Protocol ambiguity + measurement quality | **Yes** — C2-28 is 3.034× on the mean, 2.923× on the median | No |

**No defect changes what C2-28 computes.** The forward path is correct and the
classifier receives a correct `[B, 3, 224, 224]` tensor in all three C2 variants.

---

# A1 — Latency criterion: primary vs secondary metric

## A1.1 What the pre-registered rule says

`docs/ARCHITECTURE_V2_BENCHMARK.md` §1:

> Preferred: params ≤ ~3 M, GFLOPs ≤ 3×, **latency ≤ 3×**.
> Hard reject: GFLOPs > 5× or **latency > 5×**.

**The rule never names a batch size.** The benchmark measures two (1 and 32), so
"latency" has two possible referents and the pre-registered text does not choose
between them.

## A1.2 What the implementation actually does — **[MEASURED]**

`scripts/benchmark_architectures.py:284`:

```python
lat = r["vs_baseline"][f"latency_ratio_bs{args.batch_sizes[0]}"]
hard_reject = fl > REJECT_FLOPS_RATIO or (reportable and lat > REJECT_LATENCY_RATIO)
```

The criterion is bound to **`args.batch_sizes[0]`** — whichever batch size happens
to be listed first on the command line.

This is worse than an ambiguity: the verdict is a function of **CLI argument
order**. The official command used `--batch-sizes 1 32`, so the recorded verdicts
were computed on **batch size 1 only**. Had it been invoked as
`--batch-sizes 32 1` — identical measurements, identical hardware — the verdict
field would have been computed on batch 32 and three candidates would have
flipped. Nothing in the JSON records which referent was used.

The pre-registered rule was therefore never fully specified, and its
operationalisation was accidental rather than chosen.

## A1.3 What the implemented rule produces on the measured numbers — **[DERIVED]**

Replicating lines 283–295 with the thresholds read from source and the officially
measured T4 values:

| Candidate | FLOPs ratio | latency ×, BS1 | latency ×, BS32 | `hard_reject` **in the JSON** | had BS32 been first |
|---|---|---|---|---|---|
| C0 = Original AE-TFPE | 88.002 | 8.477 | 47.033 | **true** | true |
| `C2` = C2-7 | 2.740 | 8.028 | 5.378 | **true** | true |
| C2-14 | 3.376 | 4.868 | 5.352 | **false** | **true** |
| C2-28 | 4.425 | 3.034 | 5.635 | **false** | **true** |

`meets_preferred` is **false** for all four, under either referent.

### Reconciliation against the raw JSON — **[MEASURED]**

`t4_benchmark.json` was read directly. Provenance: commit
`3d5f1c424da9107b321bb304a04d6f7db8b31d74`, clean tree, Tesla T4, torch
2.11.0+cu128, `device_audit` present and `thop_buffers_removed: 90` — i.e. the run
used the device-placement fix, and every candidate was verified on-device.

**The audit brief reported `C2-7 hard_reject=false`; the JSON actually records
`hard_reject: true` for that row.** The implementation, the replication above and
the JSON all agree. There is no additional defect in verdict assembly.

One likely source of the confusion, and a hazard in its own right: **the JSON
names the 7×7 candidate `C2`, while every document calls it `C2-7`.** The key in
`CANDIDATES` is `"C2"`; `"C2-7"` exists only in prose. Any script or reader
matching on `C2-7` finds nothing, and `C2` sorts adjacent to `C2-14` / `C2-28`.
Recommend renaming the key to `C2-7` in a future run and stating the alias
wherever the JSON is quoted.

**The genuine inconsistency stands, and it is the one that matters:** C2-14 and
C2-28 record `hard_reject: false` while exceeding 5× at batch 32 (5.352× and
5.635×). That is not a bug in the arithmetic — it is the unstated batch-size
binding described above.

## A1.4 Proposed clarification

Split the single unqualified "latency" criterion into two named metrics:

| Role | Metric | Used for |
|---|---|---|
| **PRIMARY DEPLOYMENT METRIC** | batch-size-1 latency | the preferred/hard-reject decision rules |
| **SECONDARY SERVER THROUGHPUT METRIC** | batch-size-32 latency and throughput | always reported; never used to reject |

**Scientific justification.** The target workflow is image-level, interactive
agricultural classification: a field or clinic operator submits one image and
waits. Under that workload the deployment-critical quantity is the time to
classify a single image, which is batch-size-1 latency. Batch-32 throughput
describes a server-side batch-processing regime that this application does not
describe, and a method cannot be rejected for a deployment mode it does not claim.

**This distinction was not stated explicitly before the results were observed.**
That is precisely why it is recorded here as an amendment rather than applied
quietly. It is a genuine clarification of an under-specified rule, not a new rule:
the implementation already behaved this way, by accident of argument order.

**Full disclosure is mandatory and non-negotiable.** Batch-32 latency and
throughput **must remain reported in every table, for every candidate**, with the
overhead stated plainly. Demoting BS32 from the *decision* rule does not license
omitting it from the *record*. The manuscript must state that Efficient AE-TFPE
carries substantial batch-throughput overhead relative to YOLOv8n-cls.

## A1.5 Effect of the clarification — stated before adoption

The amendment **changes the outcome for two candidates**, and this must be
acknowledged openly rather than buried:

| Candidate | Under BS32 as the criterion | Under BS1 as the criterion (proposed) |
|---|---|---|
| C0 | hard reject | hard reject — *unchanged* |
| C2-7 | hard reject | hard reject — *unchanged* |
| **C2-14** | **hard reject** | not rejected — **changed** |
| **C2-28** | **hard reject** | not rejected — **changed** |

The amendment is what keeps C2-28 alive as a candidate. That is exactly why it
needs a signature, a date and a justification that stands on the application
domain rather than on the result. **It does not make C2-28 "preferred"** —
`meets_preferred` remains **false** under every accounting in this document.

## A1.6 Proposed code change — **NOT APPLIED**

```diff
+# The deployment-critical metric for image-level interactive inference.
+# Pre-registered explicitly so the verdict cannot depend on CLI argument order.
+PRIMARY_LATENCY_BATCH_SIZE = 1
...
-            lat = r["vs_baseline"][f"latency_ratio_bs{args.batch_sizes[0]}"]
+            lat = r["vs_baseline"].get(f"latency_ratio_bs{PRIMARY_LATENCY_BATCH_SIZE}")
```

and record `"primary_latency_batch_size"` plus
`"secondary_metrics_reported_not_used_for_rejection"` in the JSON verdict block.

**Until this is applied, the `verdict` field in `t4_benchmark.json` is
order-dependent and must not be quoted in the manuscript.** The underlying
measurements are unaffected and remain valid.

---

# A2 — `thop` over-counts transposed convolutions by stride²

## A2.1 The defect

`thop` scores `nn.ConvTranspose2d` with the same routine it uses for `nn.Conv2d`:

```python
total_ops = y.nelement() * (m.in_channels // m.groups * kernel_ops + bias_ops)
```

For a *forward* convolution, MACs scale with the **output** element count. For a
*transposed* convolution the computation scatters from the input, so MACs scale
with the **input** element count. With stride 2 the output has 4× the elements of
the input, so every transposed convolution is over-counted by **stride² = 4×**.

## A2.2 Empirical ground truth — **[MEASURED]**

Weights and input set to 1 and bias to 0, so each output element equals the number
of multiply-accumulates that actually landed in it; `output.sum()` is then a direct
count of the MACs performed:

| Layer (`k=4, s=2, p=1`) | Empirical `output.sum()` | Analytic `Cin·H·W·Cout·k²` | `thop` |
|---|---|---|---|
| 1→1 @ 2×2 | **36** | 64 | 256 |
| 48→32 @ 56×56 | **75,700,224** | 77,070,336 | 308,281,344 |
| 64→48 @ 28×28 | **37,171,200** | 38,535,168 | 154,140,672 |
| 32→3 @ 112×112 | **19,095,936** | 19,267,584 | 77,070,336 |
| 128→64 @ 28×28 | **99,123,200** | 102,760,448 | 411,041,792 |

The analytic formula is within ~2 % of empirical (the difference is border taps
discarded by `padding=1`). `thop` is 4.0× the analytic figure and ~4.1× the
empirical one. **`thop` is wrong; the input-scatter formula is right.**

## A2.3 Effect on every candidate — **[DERIVED]**

Correction = `thop` total − `thop`'s transposed-conv contribution + the analytic
contribution. It applies **uniformly to every candidate containing an
auto-encoder**, including Original AE-TFPE.

| Candidate | GFLOPs as reported | × base | GFLOPs corrected | × base | Over-count removed |
|---|---|---|---|---|---|
| BASELINE | 0.4116 | 1.000 | 0.4116 | 1.000 | 0.0000 |
| **C0 = Original AE-TFPE** | 36.2215 | **88.00×** | 34.8728 | **84.73×** | 1.3487 |
| C2-7 | 1.1279 | 2.740 | 0.9786 | **2.378** | 0.1493 |
| C2-14 | 1.3896 | 3.376 | 1.0043 | **2.440** | 0.3854 |
| **C2-28** | 1.8215 | **4.425** | 1.0123 | **2.459** | 0.8092 |

## A2.4 Why this matters, and why it is not result-fitting

The correction **improves C2-28's FLOPs ratio from 4.425× to 2.459×**, i.e. from
outside to inside the ≤3× preferred band. Stated plainly so it cannot be
overlooked. Three things constrain the temptation:

1. **It changes no verdict.** `meets_preferred` for C2-28 stays **false**, because
   BS1 latency is 3.034× against a 3.0× threshold — it fails by 0.034× regardless
   of FLOPs. The correction rescues nothing on the primary axis.
2. **It applies uniformly and is not selective.** It also reduces C0's headline
   figure from 88.00× to 84.73×, and it *flattens* the spatial-resolution FLOPs
   story that `ARCHITECTURE_V2_SPATIAL_TRADEOFF.md` relies on: the C2-7/14/28
   progression was 2.74 → 3.38 → 4.43 and is really **2.38 → 2.44 → 2.46**, i.e.
   near-flat. FLOPs barely separate the three grids; measured BS1 latency
   (8.03× / 4.87× / 3.03×) separates them decisively. Any claim resting on the
   FLOPs progression must be rewritten.
3. **It is verifiable.** The empirical table above is reproducible in seconds and
   does not depend on trusting either implementation.

## A2.5 Recommendation — **NOT APPLIED**

Report **both** columns until a decision is taken. If adopted, the corrected
convention must be applied to every FLOPs figure in the manuscript, including
Original AE-TFPE's, and the 88.00× headline becomes 84.73×. Adopting it for the
Efficient method only would be indefensible.

---

# A3 — `SlimFeatureSpaceAE.describe()` reports the wrong decoder depth

## A3.1 The defect — **[MEASURED]**

`src/aetfpe/autoencoder/model.py:195`:

```python
"decoder": f"{len(self.decoder) // 2} x [ConvT4x4 s2], {self.grid} -> {self.out_size}",
```

`self.decoder` holds `len(decoder_widths)` `_up_block` modules **plus** a final
bare `ConvTranspose2d` **plus** a `Sigmoid`, so `len(self.decoder) // 2` has no
relation to the number of upsampling stages. The correct count is
`len(decoder_widths) + 1`.

| Candidate | `describe()` string | Actual upsampling stages | Actual path |
|---|---|---|---|
| C2-7 | "3 x [ConvT4x4 s2], 7 -> 224" | **5** | 7→14→28→56→112→224 |
| C2-14 | "2 x [ConvT4x4 s2], 14 -> 224" | **4** | 14→28→56→112→224 |
| C2-28 | "2 x [ConvT4x4 s2], 28 -> 224" | **3** | 28→56→112→224 |

The **string is wrong; the architecture is right.** Every variant reaches 224×224
by pure learned transposed convolution, which is why `classifier_input` is
correctly `[B, 3, 224, 224]`.

## A3.2 Proposed fix — **NOT APPLIED**

```diff
-            "decoder": f"{len(self.decoder) // 2} x [ConvT4x4 s2], {self.grid} -> {self.out_size}",
+            "decoder": (f"{len(self.decoder) - 1} x [ConvT4x4 s2 (BN+ReLU except last)], "
+                        f"{' -> '.join(str(self.grid * 2 ** i) for i in range(len(self.decoder)))}"),
```

Not applied because it alters recorded benchmark metadata. **The already-produced
`t4_benchmark.json` carries the incorrect string**, so no manuscript table or
figure may be built from that field until this is resolved. Use §B of the decoder
audit below instead.

---

# A4 — Dead parameters in every C2-* candidate

## A4.1 The defect — **[MEASURED]**

`TimmGlobalContextRGB` builds `project` (1×1 Conv to 3 channels) and `norm`
(BatchNorm2d(3)) for the image-space output contract. In the feature-space path
(`ae_space="feature"`) the front-end calls `self.tf.forward_features(pe)`, which
**bypasses `forward()` entirely**, so `project` and `norm` are never executed.

A backward pass confirms it — these parameters receive `grad is None`:

| Candidate | Dead tensors | Dead values | Share of total params |
|---|---|---|---|
| C2-7 | 4 | 969 | 0.038 % |
| C2-14 | 4 | 201 | 0.010 % |
| **C2-28** | 4 | **153** | **0.009 %** |

`['tf.project.weight', 'tf.project.bias', 'tf.norm.weight', 'tf.norm.bias']`

## A4.2 Assessment

This is the **same class of defect as the dead-fusion bug** that G1's
backward-pass check caught in `A5_aetfpe_full` (see `aetfpe.py`, the `self.fusion
= None` comment). Consequences:

- **No effect on the forward computation**, on training dynamics, or on any
  accuracy result. The modules are not called; optimisers skip `None` gradients.
- **The reported parameter count is inflated by 153** for C2-28. The measured
  1,716,739 is therefore 1,716,586 real parameters plus 153 dead ones.
- **A G1-style backward check would fail** for every C2-* arm, exactly as it did
  for the dead-fusion bug.

## A4.3 Proposed fix — **NOT APPLIED**

Do not build `project`/`norm` when the encoder is used in feature-space mode
(mirroring the `self.fusion = None` precedent). Not applied because it changes a
freshly measured, officially recorded parameter count, and because the audit brief
forbids redesign. **It does not block the C2-28 sanity training**, whose result
would be bit-identical either way — but it must be resolved before any parameter
count is published.

---

# A5 — The latency rule does not specify mean or median, and at batch 1 it matters

## A5.1 The measurement is noisy at batch 1 — **[MEASURED]**

Read from `t4_benchmark.json`. Batch-1 timings carry 15–30 % relative standard
deviation; batch-32 timings are an order of magnitude tighter.

| Candidate | BS1 mean | BS1 median | BS1 std | std/mean | × base (mean) | × base (median) |
|---|---|---|---|---|---|---|
| BASELINE | 3.3632 | 3.2168 | 0.4881 | 14.5 % | 1.000 | 1.000 |
| C0 | 28.5088 | 23.5584 | 8.5585 | **30.0 %** | 8.477 | 7.324 |
| `C2` = C2-7 | 27.0000 | 28.5828 | 6.7475 | 25.0 % | 8.028 | 8.885 |
| C2-14 | 16.3708 | 14.1832 | 4.4528 | 27.2 % | 4.868 | 4.409 |
| **C2-28** | 10.2045 | 9.4031 | 2.0407 | 20.0 % | **3.034** | **2.923** |

| Candidate | BS32 mean | BS32 median | std/mean | × base (mean) | × base (median) |
|---|---|---|---|---|---|
| BASELINE | 7.3301 | 7.3019 | 1.5 % | 1.000 | 1.000 |
| C0 | 344.7599 | 343.5652 | 2.7 % | 47.033 | 47.051 |
| `C2` = C2-7 | 39.4217 | 39.3787 | 2.0 % | 5.378 | 5.393 |
| C2-14 | 39.2333 | 39.1881 | 0.9 % | 5.352 | 5.367 |
| C2-28 | 41.3020 | 41.3051 | 0.9 % | 5.635 | 5.657 |

## A5.2 Why this is decision-relevant

The pre-registered rule says "latency ≤ 3×" without stating **which statistic**.
The harness records mean, median and std, and the verdict uses the **mean**.

**On the mean, C2-28 is 3.034× — outside the preferred band. On the median it is
2.923× — inside it.** The band membership of the leading Efficient candidate is
decided by an unstated choice of summary statistic, on a measurement whose
standard deviation (2.04 ms) is **60× larger than the gap to the threshold**
(0.034× ≈ 0.11 ms).

The effect is systemic, not specific to C2-28: C0 moves 8.477 → 7.324 and C2-14
moves 4.868 → 4.409. Only `C2`/C2-7 moves the other way (8.028 → 8.885).

## A5.3 Recommendation — **do not choose the statistic that helps**

Switching to the median now would move C2-28 inside the preferred band, which is
precisely why it must **not** be done as a post-hoc choice. The defensible options are:

1. **Re-measure batch-1 latency under a tighter protocol** — more timed
   iterations, longer warm-up, locked GPU clocks, and an idle session — then
   pre-register mean-or-median *before* reading the result. This is the
   recommended path; it costs one short T4 run.
2. Report **both** statistics with the standard deviation everywhere, and treat
   any candidate whose mean and median straddle a threshold as **indeterminate on
   that criterion** rather than pass or fail.

Under option 2, C2-28's BS1 latency criterion is **INDETERMINATE**, which is
weaker than the "CONDITIONAL" classification used elsewhere in this document and
should be read as such until re-measured.

**No claim in the manuscript may depend on the 3× boundary until batch-1 latency
is re-measured with lower variance.** The qualitative ordering
(C2-28 ≪ C2-14 < C2-7 ≈ C0 at batch 1) is robust across mean and median and is
safe to state.

---

# Decisions required

| # | Decision | Blocks |
|---|---|---|
| 1 | ~~Supply the raw C2-7 `verdict` block~~ — **resolved**, §A1.3: the JSON records `true`, matching the implementation | — |
| 2 | Adopt BS1-primary / BS32-secondary, with BS32 always reported? | Any Efficient-side training claim |
| 3 | Adopt corrected transposed-conv FLOPs accounting, applied to **all** candidates? | Every FLOPs figure, incl. the 88.00× headline |
| 4 | Apply the `describe()` fix and re-record metadata? | Manuscript architecture table |
| 5 | Remove the dead `project`/`norm` and re-measure params? | Published parameter counts |
| 6 | **Re-measure batch-1 latency with lower variance** (§A5), pre-registering mean-or-median first | Any claim resting on the 3× boundary |
| 7 | Rename the JSON candidate key `C2` → `C2-7` (§A1.3) | Anyone matching the JSON on documented names |

Items 3–5 and 7 are one-line changes. Item 6 costs one short T4 run. **None is
applied here.**

**None of these blocks the single C2-28 clean-validation sanity run**, which
measures whether the 28×28 representation is learnable — a question none of the
four items affects.
