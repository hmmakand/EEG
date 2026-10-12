"""Accuracy, balanced accuracy, macro F1 and the confusion matrix."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def classification_metrics(labels, predictions) -> dict:
    """Class order [0, 1] = [left, right]; matrix rows are true classes, columns predictions."""
    if len(labels) == 0 or len(labels) != len(predictions):
        raise ValueError("Labels and predictions must be non-empty and of equal length")
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    return dict(accuracy=float((labels == predictions).mean()),
                balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
                macro_f1=float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
                confusion_matrix=confusion_matrix(labels, predictions, labels=[0, 1]).tolist())


def summarize(subject_results: dict) -> dict:
    """Mean of the per-subject scores and the summed confusion matrix."""
    values = list(subject_results.values())
    return dict(accuracy=float(np.mean([r["accuracy"] for r in values])),
                balanced_accuracy=float(np.mean([r["balanced_accuracy"] for r in values])),
                macro_f1=float(np.mean([r["macro_f1"] for r in values])),
                confusion_matrix=np.sum([r["confusion_matrix"] for r in values], axis=0).tolist())


SCORES = ("final_epoch", "highest_epoch")


def summarize_both(subject_results: dict) -> dict:
    """`summarize` for the final-epoch scores and for the highest-epoch scores."""
    return {score: summarize({subject: result[score] for subject, result in subject_results.items()})
            for score in SCORES}
