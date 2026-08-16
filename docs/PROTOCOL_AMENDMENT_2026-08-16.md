# Protocol Amendment Record — 16 August 2026

**Status:** **ADOPTED 16 Aug 2026.** A1, A2 and A4 are applied in code; A5 is
decided (statistic held, measurement to be repeated); A3 remains open. The raw
benchmark JSON has **not** been edited — it is preserved verbatim at
`docs/evidence/2026-08-16_t4_benchmark_raw.json`, sha256
`ecaf7071b8993338db1f00dcbf19474fa5c88c039ef97d7da211c72eebf4e39a`.

**Trigger:** the official Tesla T4 benchmark (first CUDA run of
`scripts/benchmark_architectures.py`) plus two commissioned audits — decision-rule
consistency, and the decoder shape path.

**Why this file exists.** Four of the five items below were discovered *after*
observing the T4 results. Any change to a decision rule, a measurement convention
or a reported figure made after seeing results must be recorded as an explicit,
dated amendment with its justification and its effect on every candidate —
otherwise it is indistinguishable from fitting the rule to the answer. Each item
therefore states what it changes, what it does **not** change, and whether it
moves any candidate across any threshold.

Evidence labels: **[MEASURED]** on real code/hardware · **[DERIVED]** arithmetic
from measurements · **[AWAITING DATA]**.

---

## Summary of the five items

| # | Item | Class | Changes a verdict? | Status |
|---|---|---|---|---|
| **A1** | Latency rule does not name a batch size | Protocol ambiguity | **Yes** — see §A1.5 | **APPLIED** — `PRIMARY_LATENCY_BATCH_SIZE = 1` |
| **A2** | `thop` over-counts `ConvTranspose2d` by stride² | Measurement-tool defect | **No** | **APPLIED** — handler overridden in `count_flops` |
| **A3** | `SlimFeatureSpaceAE.describe()` reports the wrong decoder depth | Documentation defect | No | **OPEN** — not applied |
| **A4** | `tf.project` / `tf.norm` are dead parameters in every C2-* candidate | Implementation defect | No | **APPLIED** — head not built in feature-space mode |
| **A5** | Latency rule does not say mean or median, and BS1 std is 15–30 % | Protocol ambiguity + measurement quality | **Yes** — 3.034× mean vs 2.923× median | **DECIDED** — mean retained, BS1 to be re-measured |

**Applied means: fixed in code, with the raw pre-correction evidence preserved
unedited.** A2 and A4 changed reported numbers but **no output**: logits,
front-end tensors and AE latents are bit-identical before and after, verified by
transferring the pre-fix weights so the comparison isolates computation from
initialisation RNG.

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

### Mandatory disclosure

**This clarification was introduced *after* the T4 results were observed, and
specifically after discovering that the previous benchmark verdict depended on the
order in which batch sizes were passed on the command line.** It was not stated
explicitly beforehand. That sequence must be disclosed wherever the decision rule
is described, including in the manuscript and in any response to reviewers.

It is a clarification of an under-specified rule rather than a new rule — the
implementation already computed the criterion on batch 1, by accident of argument
order rather than by choice — but the fact that it was pinned down after seeing
which candidates it favoured is exactly what the reader is entitled to know.

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

## A1.6 Code change — **APPLIED**

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

The verdict block now also records `primary_latency_batch_size`,
`primary_latency_statistic`, `primary_latency_ratio` and
`secondary_metrics_reported_not_used_for_rejection`, so the referent is explicit
in every future run.

**The `verdict` field in the preserved raw JSON remains order-dependent** — it was
produced before this fix — and must not be quoted. Its underlying measurements are
unaffected and remain valid.

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

## A2.5 Resolution — **APPLIED as a measurement-tool defect fix**

`count_flops` now passes a `custom_ops` handler that counts transposed
convolutions from input elements, overriding thop's. Applied **uniformly to every
candidate**: Original AE-TFPE's headline becomes **84.73×**, not 88.00×. Applying
it to the Efficient method only would have been indefensible.

Authoritative figures regenerated with the fixed tool (which also reflect the A4
parameter cleanup):

| Candidate | Params | × base | GFLOPs | × base | Model size |
|---|---|---|---|---|---|
| BASELINE | 1,488,247 | 1.000 | 0.4116 | 1.000 | 5.703 MB |
| C0 = Original AE-TFPE | 87,549,123 | 58.827 | 34.8731 | 84.726 | 334.576 MB |
| C1 | 2,700,147 | 1.814 | 1.6602 | 4.034 | 10.919 MB |
| `C2` = C2-7 | 2,544,634 | 1.710 | 0.9789 | 2.378 | 10.325 MB |
| C2-14 | 2,066,890 | 1.389 | 1.0046 | 2.441 | 8.495 MB |
| **C2-28** | **1,716,586** | **1.153** | **1.0126** | **2.460** | **7.155 MB** |
| C3 | 2,263,930 | 1.521 | 0.6565 | 1.595 | 9.251 MB |

One consequence must be stated plainly: **C1 was never hard-rejected.** At 4.03×
it exceeds the ≤3× preferred band but clears the 5× threshold, so the earlier
"C1 is hard-rejected at 7.31×" claim was an artifact of the defect and is
withdrawn. C1 has no measured latency and is not reinstated as a candidate.

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

## A4.3 Fix — **APPLIED as an implementation-defect cleanup**

`TimmGlobalContextRGB` takes `image_space_head`; `AETFPE` passes
`image_space_head=False` whenever `use_ae and ae_space == "feature"`, so the
projection head is not built on the path that never calls it. `forward()` raises
if invoked without a head, so the dead parameters cannot silently return. This
mirrors the `self.fusion = None` precedent and is **not an architecture redesign**:
no executed operation changed.

Gated on proof, via `scripts/prove_dead_parameters.py`, which requires three
independent tests to agree — the owning module never executes, the parameter is
absent from the autograd graph reachable from the logits, and it receives no
gradient. The controls are what make this safe: **C0 and C1 have zero dead
parameters** and correctly keep their heads.

| Candidate | Params before | Params after | Removed | Trainable after |
|---|---|---|---|---|
| C2-7 | 2,545,603 | 2,544,634 | 969 | 1,593,610 |
| C2-14 | 2,067,091 | 2,066,890 | 201 | 1,575,546 |
| **C2-28** | 1,716,739 | **1,716,586** | **153** | 1,567,066 |
| C3 | 2,264,323 | 2,263,930 | 393 | 1,581,322 |
| C1 / C0 *(controls)* | unchanged | unchanged | **0** | unchanged |

**Outputs verified bit-identical** for C2-7 / C2-14 / C2-28 / C1 — logits,
front-end tensor and AE latent — by loading the pre-fix weights into the post-fix
model, so the comparison tests computation rather than initialisation RNG.
Forward/backward re-validated afterwards: finite logits, finite gradients, and
**zero dead parameters in every candidate**.

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

## A5.3 Decision — **the mean is retained; the measurement is repeated**

Switching to the median would move C2-28 inside the preferred band, which is
precisely why it is **not** done. `PRIMARY_LATENCY_STATISTIC = "latency_ms_mean"`
is pinned in code. The measurement is re-run instead, at
warm-up 100 / 1000 timed iterations / batch 1, reporting mean, std, median and
p95 — with the decision statistic fixed in advance as the mean. The options
considered were:

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

| # | Decision | Status |
|---|---|---|
| 1 | Supply the raw C2-7 `verdict` block | **RESOLVED** — §A1.3: the JSON records `true`, matching the implementation |
| 2 | Adopt BS1-primary / BS32-secondary, BS32 always reported | **ADOPTED & APPLIED** — `PRIMARY_LATENCY_BATCH_SIZE = 1` |
| 3 | Adopt corrected transposed-conv FLOPs accounting for **all** candidates | **ADOPTED & APPLIED** — the 88.00× headline is now 84.73× |
| 4 | Apply the `describe()` fix and re-record metadata | **OPEN** — one line; the true decoder path is recorded in `DECODER_PATH_AUDIT.md` meanwhile |
| 5 | Remove the dead `project`/`norm` and re-measure params | **ADOPTED & APPLIED** — gated on three-way proof; outputs bit-identical |
| 6 | Re-measure batch-1 latency with lower variance, statistic fixed first | **PENDING** — statistic fixed to the **mean**; run prepared, warm-up 100 / 1000 iters / batch 1 |
| 7 | Rename the JSON candidate key `C2` → `C2-7` | **OPEN** — cosmetic; noted wherever the JSON is quoted |

Only A3 (item 4) and the key rename (item 7) remain unapplied, and neither affects
any measurement.

**None of these blocks the single C2-28 clean-validation sanity run**, which asks
whether the 28×28 representation is learnable — a question no item here affects.


