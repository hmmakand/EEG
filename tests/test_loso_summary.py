from __future__ import annotations

import pytest

from scripts.braindecode_scripts.train_loso import _summarize_results


def test_loso_summary_uses_matching_row_weights_when_metrics_are_missing() -> None:
    rows = [
        {"n_test_windows": 10, "test_accuracy": 0.1},
        {"n_test_windows": 100},
        {"n_test_windows": 30, "test_accuracy": 0.9},
    ]

    summary = _summarize_results(rows)

    assert summary["test_accuracy_mean"] == pytest.approx(0.5)
    assert summary["test_accuracy_weighted"] == pytest.approx(0.7)
