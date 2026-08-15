# Evaluation Protocols

**Question:** the recovered historical evaluation was

```python
model.val(split="test", save_json=True, max_det=1, conf=0.25, imgsz=224, device="mps")
```

with a custom script that counted images without a prediction as errors. Is that
semantically appropriate for YOLO image classification?

**Answer: no. It is a detection-style evaluation, and it cannot have been run by
a classification model at all.**

---

## 1. What the evidence shows

### 1.1 The Ultralytics classification validator does not accept those arguments

Inspecting `ultralytics/models/yolo/classify/val.py` in the installed
distribution:

- there is **no confidence threshold** anywhere in the classification path;
- there is **no `max_det`**;
- there is **no `save_json`**, no `pred_to_json`, no `eval_json`;
- top-1/top-5 come from `torch.argsort` over the full logit vector, for **every**
  image in the split.

So `conf` and `max_det` are silently ignored by a classification run, and
`save_json=True` produces nothing. A classification `val()` cannot emit
`best_predictions.json`.

### 1.2 The archived predictions are detection output

Every record in the four archived `best_predictions.json` files has the shape

```json
{"image_id": "Background_without_leaves_image__880_", "category_id": 21,
 "bbox": [3.929, 0.0, 250.071, 192.0], "score": 0.37158}
```

`bbox` and `score` are detection fields. The runs also produced `PR_curve.png`,
`P_curve.png`, `R_curve.png`, `F1_curve.png` and `confusion_matrix.png` — the
detection validator's artifact set. The datasets were
`yolo_labels_lant_leaf_disease_*` with `labels/test/*.txt`, i.e. YOLO detection
format with whole-image boxes.

### 1.3 The accuracy definition that was actually used

`calculate_top1.py`:

```python
label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]
for label_file in label_files:
    true_category = get_true_label(...)          # class id from the .txt
    predicted_category = None
    for prediction in predictions:               # from best_predictions.json
        if prediction["image_id"] == image_id:
            predicted_category = prediction["category_id"]; break
    if predicted_category is not None and predicted_category == true_category:
        correct_predictions += 1
    total_labels += 1
top_1_accuracy = correct_predictions / total_labels
```

So the historical metric is

```
Acc_legacy = |{ images whose single highest-scoring box, with score >= 0.25,
                 has the correct class }| / |{ all label files }|
```

An image whose best box falls below `conf=0.25` contributes **no** entry to the
JSON and is scored as an error. In the archived baseline runs this is not a
corner case: only 4,832 of 8,340 images produced any detection at all — **42% of
the test set was rejected before classification was even considered.**

### 1.4 Consequences

| Issue | Consequence |
|---|---|
| Confidence rejection | `Acc_legacy ≤ classification top-1`. The metric conflates "predicted the wrong class" with "was not confident enough to predict". |
| `max_det=1` | **Top-5 is undefined.** One prediction per image cannot yield a top-5 figure. |
| Detection thresholding interacts with corruption | Corruption lowers confidence as well as accuracy, so `Acc_legacy` degrades faster than true top-1 — and *differently* for models with different confidence calibration. Two models can swap rank purely through calibration. |
| Mixed pipelines in one paper | `log-org-280223` is a genuine **classification** run (`task=classify`, top1/top5). The YOLOv6/7/9/10 results are **detection**. Tables 2–3 report Top-5, which the detection path cannot produce. **Tables 2–3 and Fig. 9 cannot both come from the same pipeline.** |

That last row is a pre-existing inconsistency in the original work. The revision
must disclose it rather than inherit it.

---

## 2. Two explicitly separated protocols

### Protocol A — LEGACY REPRODUCTION

**Purpose:** reproduce historical manuscript numbers, and nothing else.

| Property | Value |
|---|---|
| Task | object detection with whole-image boxes |
| Validator | Ultralytics detection `val()` |
| Arguments | `max_det=1, conf=0.25, imgsz=224, save_json=True` |
| Metric | `Acc_legacy` as defined in §1.3 — includes confidence rejection |
| Top-5 | **not available** |
| Denominator | count of label files, including undetected images |
| Split | whatever the historical run used (evidence points to `val`) |

**Status:** not implemented, and not planned. Reproducing it would require the
`yolo_labels_*` detection-format datasets and the corrupted sets, all of which
are gone. Protocol A is documented here so the historical numbers can be
*described* correctly, not recomputed.

**Use:** citation only. Any historical figure quoted in the revision must be
labelled "Protocol A (legacy, detection-style, includes confidence rejection)".

### Protocol B — REVISED CLASSIFICATION *(this framework)*

**Purpose:** every new result in the revised manuscript.

| Property | Value |
|---|---|
| Task | image classification |
| Model | front-end → stock YOLOv8n-cls (or baseline backbone) |
| Metric | standard top-1 and top-5 over the full logit vector |
| Confidence rejection | **none** — every image receives a prediction |
| Denominator | every image in the evaluated directory |
| Additional | per-class precision / recall / F1, confusion matrix, macro averages |
| Split | frozen `test` split, plus the frozen corrupted copies of it |
| Implementation | `src/aetfpe/metrics.py::summarize`, called by `scripts/evaluate.py` |

Every image is scored; `num_images` in each result equals the directory's file
count, which is machine-checkable against the corruption manifest.

---

## 3. Which protocol for which table

| Table | Protocol | Why |
|---|---|---|
| Component ablation (A0–A5) | **B** | Arms differ only in the front-end; a confidence threshold would add a second, uncontrolled axis of variation |
| Fusion comparison (D1, F1, F2, F4) | **B** | Fusion operators change logit calibration; Protocol A would let a calibration shift masquerade as a robustness difference |
| Mechanism gate (M1, M2, M3) | **B** | Must be commensurable with A0 and A5 |
| Corruption benchmark | **B** | Corruption depresses confidence independently of correctness; Protocol A would confound the two and is the single worst place to use it |
| External baselines (ResNet-50, EfficientNet-B0, ViT-B/16) | **B** | These are classifiers; they have no detection head, so Protocol A is not even definable for them |
| Historical Tables 2–3, Figs. 8–12 | **A**, cited as legacy | Cannot be recomputed; label them explicitly |

**Protocol B is used for every new table.** The reason is not merely convention:

1. **It is the only protocol the baselines admit.** ResNet-50 and ViT-B/16 have
   no detection head. A fair comparison across all sixteen arms requires a metric
   all sixteen can produce.
2. **It removes a confound from the paper's central claim.** The paper is about
   robustness under degradation. Degradation lowers confidence *and* accuracy.
   Under Protocol A those two effects are summed into one number, so a model that
   is merely better-calibrated under noise looks more robust. Protocol B measures
   the classification claim the paper actually makes.
3. **Top-5 becomes definable**, restoring the metric Tables 2–3 report and
   Protocol A cannot supply.
4. **It matches `log-org-280223`**, the one historical run that was genuinely a
   classification run — so the new baseline is comparable to the strongest
   surviving historical evidence.

---

## 4. Required manuscript disclosure

The revision must state, in §4.5:

> Results in the revised manuscript use standard image-classification top-1 and
> top-5 accuracy computed over all test images, with no confidence threshold.
> Figures reported in the previous version for YOLO variants were obtained from a
> detection-style evaluation (`max_det=1`, `conf=0.25`) in which images without a
> detection above threshold were counted as errors. The two are not directly
> comparable, and the earlier numbers are reported as legacy measurements where
> they are retained at all.

Silently swapping the metric would change every number in the paper without
explanation and would be, correctly, read as a discrepancy by the reviewers who
already flagged reproducibility.

---

## 5. Effect on the comparison against historical results

Protocol B is expected to give **higher** absolute accuracies than Protocol A on
the same model and data, because the ~42% of images rejected by the confidence
threshold are returned to the pool and some fraction of them are classified
correctly. Therefore:

- Do **not** claim that the new pipeline "improves" on the historical numbers.
  Part of any increase is the metric change, not the method.
- Report new results as a self-contained set under Protocol B, with the
  ablation, fusion and mechanism comparisons made **within** that set.
- Where a historical number is retained, mark it Protocol A and do not compute
  ratios or improvements across the two protocols.
