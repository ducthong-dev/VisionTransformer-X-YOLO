# Reviewer Evidence Map

**Date:** 2 September 2026 · **Status:** structure frozen before test results were
inspected. Every "AWAITING DATA" cell is filled from the evaluation outputs, whatever
they say.

Labels: **[MEASURED]** · **[DERIVED]** · **[AWAITING DATA]** · **[BLOCKED]** ·
**[NOT TESTABLE]** with the current evidence.

---

## 1. Which experiment answers which reviewer point

| Reviewer point | Evidence | Status |
|---|---|---|
| **#10.4** mechanism attribution — is the effect the architecture or the historical LUT? | M1 (legacy LUT), M2 (gamma), M3 (augmentation) vs E5 | **[MEASURED]** on validation — **E5 is below all three** (see §3) |
| **#10.4** AE contribution, Original side | A5 vs A3 | **[AWAITING DATA]** — test |
| **#10.4** AE contribution, Efficient side | E5 vs E3 | **[AWAITING DATA]** · **CONFOUNDED** — E3 also changes the fusion space (grid → image) |
| **#10.4** denoising objective | **A5 vs D1** — identical architecture, objective differs | **[AWAITING DATA]** — the cleanest available test, and the headline of the controlled benchmark |
| **#10.6** efficiency of the proposed method | T4 benchmark: 1,716,586 params, 1.0126 GFLOPs, 51.0× fewer params and 34.4× fewer FLOPs than Original | **[MEASURED]** — Tesla T4, reported in its own labelled table |
| **#10.6** architecture justification for the 28×28 grid | E7 (C2-7) = 0.84364 vs E5 = 0.98778 best val top-1 | **[MEASURED]** — supports retaining 28×28 **over the tested 7×7 control**; does not establish 28×28 as globally optimal (no 14×14 control) |
| **#10.7** fairness — identical protocol for every arm | 16 runs, one frozen protocol, one seed, one dataset hash | **[MEASURED]** — `docs/RUN_INVENTORY.md` |
| **#10.7** fair external comparison | B2 (EfficientNet-B0) = 0.99880, the **best** clean arm | **[MEASURED]** validation; **[AWAITING DATA]** test |
| **#10.7** further external baselines | B1 (ResNet-50) | **[BLOCKED]** — not yet run; skipped by the 20 M-parameter budget policy |
| **#10.7** ViT-B/16 external baseline | B3 | **[BLOCKED]** — force-run 2026-09-01T18:04, **stalled at 2/50**; resumable, ≈4 A100-h to finish |
| **#11** robustness benchmark | Clean/Easy/Moderate/Hard | **[AWAITING DATA]** — *synthetic augmentation* robustness only (§2) |
| **#11** robustness attribution vs plain augmentation | M3 vs E5 | **[AWAITING DATA]** |
| **#12** component contribution (PE / TF / AE) | **A0 → A1 → A2 → A3 → A4 → A5**, plus E3 | **[AWAITING DATA]** — the ablation sequence is now **complete**: A2 finished 50/50 on 2026-09-01 |
| **#12** fusion comparison | A5 vs F1 / F2 / F4 | **[AWAITING DATA]** · F2 **CONFOUNDED** — the only arm that modifies the classifier stem |
| **#10** "noise-resilient latent features" | **Controlled Synthetic Corruption Benchmark** | **[AWAITING DATA]** — see §2; the Easy/Moderate/Hard sets **cannot** answer this |

---

## 2. The two robustness benchmarks — never merged

| | Benchmark | What it measures | Can it answer the noise challenge? |
|---|---|---|---|
| **A** | Clean / Easy / Moderate / Hard | synthetic **augmentation** robustness — geometry, photometry, occlusion and noise entangled | **No.** ~84 % of `hard` is flipped or rotated (`docs/CORRUPTION_PROTOCOL.md` §4) |
| **B** | **Controlled Synthetic Corruption Benchmark** | targeted **corruption / noise** robustness — 6 non-geometric families × mild/moderate/severe | **Yes**, for corruption robustness specifically |

B was added **after** the corruption audit revealed the geometric confound, but **before**
any model's test performance was inspected. It is a frozen specification, **not a
pre-registration**, and it does **not** replace A. Never called a *real-world robustness
benchmark*. Full rationale: `docs/GATE1_RESOLUTION_AND_ANALYSIS_PROTOCOL.md` §6.

---

## 3. What the reviewers must be told regardless of the test results

These are already established and do not depend on evaluation:

1. **GATE 1 was not passed.** The pre-registered gate required E5 to exceed the best of
   M1/M2/M3 on clean validation. E5 = 0.98778; M2 = 0.99760, M3 = 0.99688, M1 = 0.99581,
   A0 = 0.99760. E5 ranks 15th of 16. **[MEASURED]**
2. **E5 is 0.98–1.10 pp below the strongest clean baselines** — outside the saturated
   cluster, roughly 2.3× its full spread. **[MEASURED]**
3. **Clean validation is saturated** — 14 of 16 arms within 0.43 pp, at one seed. Small
   ranking differences there are not robust evidence of architectural superiority.
4. **Single seed.** No multi-seed claim. Paired statistics quantify test-sample
   uncertainty only.
5. **The best clean arm is an off-the-shelf EfficientNet-B0**, not the proposed method.
6. **Two external baselines are absent.** B1 (ResNet-50) was never run. B3 (ViT-B/16) — the
   very backbone the Original method builds on — was force-run but stalled at epoch 2/50
   and is excluded; it is resumable in ≈4 A100-hours.
7. **The component ablation is complete.** A2 finished 50/50 (best val top-1 0.9964055)
   and is included; no arm in the A0–A5 sequence is missing.
8. **Three controls are confounded** and are not presented as one-variable experiments:
   E3 (AE + fusion space), F2 (classifier stem), and the Easy/Moderate/Hard tiers
   (geometry + corruption).
9. **The Easy/Moderate/Hard corruption sets are unseeded and not regenerable**; the saved
   PNGs are the only record, and the Easy/Moderate generating parameters are lost.
10. **`configs/_base.yaml` still declares 30 epochs**; all runs used 50 via override. The
    file is deliberately not back-edited; the effective config is documented.

---

## 4. Filled from the evaluation, in this order

1. `docs/CONSOLIDATED_EVIDENCE_TABLE.md` — 22 logical IDs × 4 distributions, validation
   and test in separately labelled columns.
2. `docs/CONTROLLED_CORRUPTION_RESULTS.md` — per-family, per-severity, with degradation
   and retention, plus paired statistics.
3. `results/evaluation/paired_statistics.json` — comparisons C1–C10.
4. `docs/MANUSCRIPT_CLAIM_AUDIT.md` — every claim SUPPORTED / CONTRADICTED / UNTESTABLE.
5. `docs/PAPER_REVISION_PLAN.md`.

If the evidence contradicts a manuscript claim, **the manuscript changes.**
