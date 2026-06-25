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
    "cv_accuracy_mean",
    "cv_accuracy_std",
    "checkpoint_path",
    "config_path",
    "history_path",
    "status",
]


def append_master_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_row(row)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MASTER_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)


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

