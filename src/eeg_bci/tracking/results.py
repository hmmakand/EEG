from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


MASTER_COLUMNS = [
    "run_id",
    "run_dir",
    "tensorboard_dir",
    "timestamp",
    "experiment",
    "dataset",
    "model",
    "subject",
    "held_out_subject",
    "seed",
    "split_strategy",
    "n_train_windows",
    "n_valid_windows",
    "n_test_windows",
    "n_chans",
    "n_outputs",
    "n_times",
    "sfreq",
    "train_loss",
    "train_accuracy",
    "valid_loss",
    "valid_accuracy",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_cohen_kappa",
    "test_macro_f1",
    "test_macro_precision",
    "test_macro_recall",
    "test_roc_auc",
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "best_score",
    "best_score_std",
    "best_train_score",
    "best_train_score_std",
    "best_params",
    "checkpoint_path",
    "config_path",
    "history_path",
    "final_metrics_path",
    "dataset_info_path",
    "subject_test_results",
    "status",
]


def master_result_path(original_cwd: Path, experiment: str, dataset: str) -> Path:
    """Return the grouped master-result CSV path."""
    safe_experiment = experiment.replace(" ", "_")
    safe_dataset = dataset.replace(" ", "_")
    return original_cwd / "outputs" / "results" / safe_dataset / f"results_master_{safe_experiment}.csv"


def append_master_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _read_existing_rows(path) if path.exists() else []
    normalized_rows = [_normalize_row(existing) for existing in existing_rows]
    normalized_rows.append(_normalize_row(row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MASTER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(normalized_rows)


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {key: "" for key in MASTER_COLUMNS}
    normalized.update({key: value for key, value in row.items() if key in normalized})
    for source, target in {
        "train_acc": "train_accuracy",
        "valid_acc": "valid_accuracy",
        "test_acc": "test_accuracy",
        "cv_acc_mean": "cv_accuracy_mean",
        "cv_acc_std": "cv_accuracy_std",
    }.items():
        if source in row and not normalized.get(target):
            normalized[target] = row[source]
    return normalized
