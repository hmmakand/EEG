from __future__ import annotations

import csv

from eeg_bci.tracking.results import append_master_result


def test_append_master_result_normalizes_legacy_keys_and_preserves_new_columns(tmp_path) -> None:
    path = tmp_path / "results_master.csv"

    append_master_result(
        path,
        {
            "run_id": "legacy",
            "test_acc": 0.25,
            "train_acc": 0.5,
            "cv_acc_mean": 0.4,
        },
    )
    append_master_result(
        path,
        {
            "run_id": "new",
            "test_accuracy": 0.75,
            "best_params": "{'optimizer__lr': 0.001}",
            "final_metrics_path": "run/metrics/final_metrics.yaml",
            "dataset_info_path": "run/metrics/dataset_info.yaml",
        },
    )

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["test_accuracy"] == "0.25"
    assert rows[0]["train_accuracy"] == "0.5"
    assert rows[0]["cv_accuracy_mean"] == "0.4"
    assert rows[1]["best_params"] == "{'optimizer__lr': 0.001}"
    assert rows[1]["final_metrics_path"] == "run/metrics/final_metrics.yaml"
    assert rows[1]["dataset_info_path"] == "run/metrics/dataset_info.yaml"
