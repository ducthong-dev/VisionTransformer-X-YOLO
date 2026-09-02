# Interim Evidence — 2 September 2026

**This is NOT the final evidence freeze.** Nine of nineteen physical runs are still
unevaluated. It is published now because the nine completed evaluations already settle
most of the scientific questions, and they do not support the manuscript's narrative.

**Evaluated on both frozen benchmarks (9):** A0 · A1 · A2 · A3 · A4 · A5 · B2 · D1 · E5
**Not yet evaluated (9):** B1 · B3 · E3 · E7 · F1 · F4 · M1 · M2 · M3  (F2 partially)

Source: `AE_TFPE_MajorRevision/evaluation/` written by the Stage B Colab session,
2026-09-01T19:04 → 2026-09-02T02:44. Frozen pipeline, unchanged. Single seed (0).

---

## 1. The headline: the auto-encoder is the entire robustness mechanism

Controlled Synthetic Corruption Benchmark, **severe** severity, Top-1:

| Model | AE? | Gauss noise | Impulse | Blur | Brightness | Contrast | JPEG |
|---|---|---|---|---|---|---|---|
| A0 baseline | no | 0.4164 | 0.3285 | 0.4424 | 0.9910 | 0.8966 | 0.4755 |
| A1 PE-only | no | 0.4181 | 0.3111 | 0.4288 | 0.9844 | 0.8687 | 0.5809 |
| A2 TF-only | no | 0.3140 | 0.4913 | 0.4260 | 0.9894 | 0.9380 | 0.4114 |
| A3 PE+TF | no | 0.2875 | 0.3200 | 0.5044 | 0.9874 | 0.9061 | 0.6481 |
| **A4 RGB+AE** | **yes** | **0.9950** | **0.9960** | 0.8575 | 0.9778 | 0.5351 | 0.9774 |
| **A5 full AE-TFPE** | **yes** | 0.9934 | 0.9945 | 0.8607 | 0.9496 | 0.5298 | 0.9833 |
| **E5 Efficient** | **yes** | 0.9867 | 0.9857 | **0.9317** | 0.8551 | 0.4012 | 0.9767 |
| B2 EfficientNet-B0 | no | 0.6779 | 0.7305 | 0.6914 | 0.9969 | 0.9863 | 0.7641 |
| D1 AE, clean objective | yes | 0.9245 | 0.6324 | 0.7802 | 0.9526 | 0.6847 | 0.9712 |

Every arm **with** an auto-encoder is transformed on additive noise; every arm **without**
one collapses. **A4 — RGB + auto-encoder, no PE, no transformer — is the single most
robust model in the campaign**, at 1.75 M parameters against A5's 87.5 M.

## 2. PE and the transformer contribute nothing

- **A1 (PE-only) ≈ A0** on every family, and *worse* on impulse noise (0.3111 vs 0.3285)
  and contrast (0.8687 vs 0.8966). The only gain is JPEG (0.5809 vs 0.4755).
- **A2 (TF-only) is worse than A0** on Gaussian noise (0.3140 vs 0.4164) and JPEG
  (0.4114 vs 0.4755); better only on impulse (0.4913) and contrast (0.9380).
- **A3 (PE+TF) is the worst arm in the campaign on Gaussian noise** (0.2875).
- On Clean/Easy/Moderate/Hard, A0/A1/A2/A3 land at 0.3728 / 0.3523 / 0.3642 / 0.3789 on
  Hard — indistinguishable, and all far below any AE arm (A4 0.6276, A5 0.5989, E5 0.5906).

**A4 ≥ A5 on nearly everything.** Adding PE and a ViT-B/16 to the auto-encoder buys no
robustness and costs 50× the parameters.

## 3. The denoising objective does add robustness — on some families, and it costs others

A5 (denoising) vs D1 (identical architecture, clean reconstruction objective) — the
cleanest available test. Paired McNemar + bootstrap, Holm-corrected:

| Distribution | A5 | D1 | Δ | Detectable |
|---|---|---|---|---|
| impulse severe | 0.9945 | 0.6324 | **+0.3621** | yes |
| impulse moderate | 0.9947 | 0.8472 | **+0.1476** | yes |
| blur severe | 0.8607 | 0.7802 | +0.0805 | yes |
| gaussian noise severe | 0.9934 | 0.9245 | +0.0689 | yes |
| jpeg severe | 0.9833 | 0.9712 | +0.0121 | yes |
| **contrast severe** | **0.5298** | **0.6847** | **−0.1549** | **yes — D1 better** |
| **contrast moderate** | 0.9506 | 0.9603 | −0.0097 | yes — D1 better |
| clean, brightness (all) | — | — | — | **no detectable difference** |

The denoising objective is real, large on noise, and **actively harmful on contrast**.

## 4. The critical negative: robustness is family-specific, not general

**Every AE arm is dramatically worse than the plain baseline on contrast degradation.**

At severe contrast: E5 **0.4012**, A5 0.5298, A4 0.5351 — against A0 0.8966, A2 0.9380,
**B2 0.9863**. E5 trails B2 by **58.5 points**. Brightness shows the same sign: E5 0.8551
vs A0 0.9910 and B2 0.9969.

The auto-encoder learns to remove *additive* corruption and, in doing so, becomes
dependent on the photometric statistics it was trained under. Any claim of general
robustness is contradicted by our own benchmark.

## 5. E5 preserves A5's noise robustness at 51× fewer parameters

| | A5 (87.5 M) | E5 (1.72 M) | Δ |
|---|---|---|---|
| gaussian noise severe | 0.9934 | 0.9867 | −0.007 |
| impulse severe | 0.9945 | 0.9857 | −0.009 |
| **blur severe** | 0.8607 | **0.9317** | **+0.071** |
| jpeg severe | 0.9833 | 0.9767 | −0.007 |
| brightness severe | 0.9496 | 0.8551 | −0.095 |
| contrast severe | 0.5298 | 0.4012 | −0.129 |
| Hard (benchmark A) | 0.5989 | 0.5906 | −0.008 |
| Clean test | 0.9948 | 0.9873 | −0.008 |

E5 matches A5 on additive noise, **beats it on blur**, and loses ground on the photometric
families where both are already weak. This is the strongest surviving contribution.

## 6. Clean accuracy: the proposed models are not the best, and it is detectable

| Arm | Clean test Top-1 |
|---|---|
| B2 EfficientNet-B0 | **0.9984** |
| A4 | 0.9974 |
| A0 / A2 | 0.9970 |
| A5 | 0.9948 |
| E5 | **0.9873** |

Paired, Holm-corrected: A5 vs A0 −0.0022 (**detectable**), E5 vs A0 −0.0097
(**detectable**), E5 vs B2 −0.0112 (**detectable**). On best-validation Top-1 the two
external baselines lead outright: **B1 0.9989216**, B2 0.9988018.

---

## 7. Preliminary status against the ten audit questions

| # | Question | Interim verdict |
|---|---|---|
| 1 | Is A5 the best classifier? | **CONTRADICTED** — B2 and A4 beat it on clean; the gap to A0 is detectable |
| 2 | Is A5 more robust than conventional baselines? | **SUPPORTED for noise/blur/JPEG; CONTRADICTED for contrast/brightness** |
| 3 | Is the AE pathway the main source of robustness? | **SUPPORTED — strongly.** A4 alone reproduces nearly all of it |
| 4 | Does the denoising objective add beyond D1? | **PARTIALLY SUPPORTED** — large on noise, negative on contrast |
| 5 | Does PE alone contribute measurable robustness? | **NOT SUPPORTED** — A1 ≈ A0, worse on two families |
| 6 | Does the transformer alone contribute? | **NOT SUPPORTED** — A2 worse than A0 on noise and JPEG |
| 7 | Does fusion choice matter? | **AWAITING DATA** — F1/F4 unevaluated, F2 partial |
| 8 | Does E5 preserve A5 robustness at lower complexity? | **SUPPORTED** — see §5 |
| 9 | Robust across all corruption families? | **CONTRADICTED** — catastrophic on contrast |
| 10 | Do B1/B2/B3 outperform the proposed models? | **PARTIALLY** — B2 wins clean, brightness, contrast; loses noise/blur/JPEG. **B1/B3 unevaluated** |

## 8. What is still required before the final freeze

| Run | Why it matters |
|---|---|
| **M1, M2, M3** | mechanism controls. **M3 (augmentation control) is the single most important missing arm** — if plain augmentation reproduces the AE's noise robustness, the contribution collapses to "train with augmentation" |
| **B1, B3** | external baselines; both now VALID. B1 has the campaign's best validation Top-1 |
| **F1, F2, F4** | fusion comparison — question 7 is unanswerable without them |
| **E3, E7** | Efficient-side AE control and the 7×7 resolution control |

The Stage B notebook resumes and will pick all nine up: the wave builder re-reads the
manifest, B1/B3 now satisfy `COMPLETED 50/50`, and completed pairs are skipped.

**No manuscript edits, no new experiments, no tuning, and no changes to any corruption
definition, seed rule, metric, statistic or selection rule were made in producing this.**
