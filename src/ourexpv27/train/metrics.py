"""Classification metrics of one subject, and their mean over subjects.

Class order [0, 1] follows DataConfig.class_names (rest/imagery, or left/right hand).
"""
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "kappa", "precision", "recall", "auc")


def classification_metrics(labels, predictions, scores=None) -> dict:
    """Accuracy, balanced accuracy, macro F1, Cohen's kappa, macro precision and recall, and the confusion matrix
    (rows are true classes, columns predictions). With `scores` (the probability of class 1), also the area under
    the ROC curve; it is None when only one class is present."""
    if len(labels) == 0 or len(labels) != len(predictions):
        raise ValueError("Labels and predictions must be non-empty and of equal length")
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    both = len(set(labels.tolist())) == 2
    return dict(accuracy=float((labels == predictions).mean()),
                balanced_accuracy=float(balanced_accuracy_score(labels, predictions)) if both
                else float((labels == predictions).mean()),
                macro_f1=float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
                kappa=float(cohen_kappa_score(labels, predictions, labels=[0, 1])) if both else 0.0,
                precision=float(precision_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
                recall=float(recall_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
                auc=float(roc_auc_score(labels, scores)) if scores is not None and both else None,
                confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist())


def summarize(subject_results: dict) -> dict:
    """Mean of every per-subject metric, and the summed confusion matrix."""
    values = list(subject_results.values())
    summary = {}
    for name in METRICS:
        present = [r[name] for r in values if r.get(name) is not None]
        summary[name] = float(np.mean(present)) if present else None
    summary["confusion_matrix"] = np.sum([r["confusion_matrix"] for r in values], axis=0).tolist()
    return summary


def format_metrics(metrics: dict) -> str:
    """One line: accuracy, balanced accuracy, macro F1, kappa, precision, recall, AUC."""
    parts = []
    for name, label in (("accuracy", "acc"), ("balanced_accuracy", "bal acc"), ("macro_f1", "F1"), ("kappa", "kappa"),
                        ("precision", "prec"), ("recall", "recall"), ("auc", "AUC")):
        value = metrics.get(name)
        parts.append(f"{label} {'  -   ' if value is None else f'{value:.4f}'}")
    return " | ".join(parts)


def print_summary(title: str, summary: dict) -> None:
    final = {name: summary.get(name) for name in METRICS}
    print(f"\n{title}")
    print(f"  final epoch (the result):              {format_metrics(final)}")
    if "highest_epoch" in summary:
        print(f"  highest test epoch (comparison only):  {format_metrics(summary['highest_epoch'])}", flush=True)
