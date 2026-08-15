"""Classification metrics: top-1/top-5, per-class precision/recall/F1, confusion matrix."""

from __future__ import annotations

import numpy as np


def topk_accuracy(logits: np.ndarray, targets: np.ndarray, ks=(1, 5)) -> dict:
    order = np.argsort(-logits, axis=1)
    out = {}
    for k in ks:
        k_eff = min(k, logits.shape[1])
        hit = (order[:, :k_eff] == targets[:, None]).any(axis=1)
        out[f"top{k}"] = float(hit.mean())
    return out


def confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (targets, preds), 1)
    return cm


def per_class_metrics(cm: np.ndarray, classes: list[str]) -> list[dict]:
    rows = []
    tp = np.diag(cm).astype(np.float64)
    pred_tot = cm.sum(axis=0).astype(np.float64)
    true_tot = cm.sum(axis=1).astype(np.float64)
    for i, name in enumerate(classes):
        precision = tp[i] / pred_tot[i] if pred_tot[i] > 0 else 0.0
        recall = tp[i] / true_tot[i] if true_tot[i] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "class_id": i,
                "class": name,
                "support": int(true_tot[i]),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "accuracy": float(recall),      # per-class accuracy == recall
            }
        )
    return rows


def macro_averages(rows: list[dict]) -> dict:
    if not rows:
        return {}
    present = [r for r in rows if r["support"] > 0]
    n = max(len(present), 1)
    return {
        "macro_precision": sum(r["precision"] for r in present) / n,
        "macro_recall": sum(r["recall"] for r in present) / n,
        "macro_f1": sum(r["f1"] for r in present) / n,
    }


def summarize(logits: np.ndarray, targets: np.ndarray, classes: list[str]) -> dict:
    preds = logits.argmax(axis=1)
    cm = confusion_matrix(preds, targets, len(classes))
    rows = per_class_metrics(cm, classes)
    out = {"num_images": int(len(targets))}
    out.update(topk_accuracy(logits, targets))
    out.update(macro_averages(rows))
    return {"overall": out, "per_class": rows, "confusion_matrix": cm.tolist()}
