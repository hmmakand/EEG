from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from eeg_bci.data.types import DatasetInfo
from eeg_bci.tracking.results import order_fieldnames
from eeg_bci.tracking.run_recording import build_master_row


def _cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "experiment_name": "exp",
            "seed": 7,
            "dataset": {"label": "ds"},
            "model": {"label": "eegnet"},
        }
    )


def _info() -> DatasetInfo:
    return DatasetInfo(n_chans=22, n_outputs=4, n_times=1000, sfreq=250.0)


def test_build_master_row_uses_run_and_hydra_dirs() -> None:
    row = build_master_row(
        _cfg(),
        {"test_accuracy": 0.5, "split_strategy": "session_train_test"},
        _info(),
        run_id="s01__ts",
        run_dir=Path("/out/subject_s01"),
        hydra_output_dir=Path("/out"),
        tb_dir=Path("/tb/s01"),
        timestamp="ts",
        subject="s01",
    )

    assert row["subject"] == "s01"
    assert "held_out_subject" not in row
    assert row["test_accuracy"] == 0.5
    assert row["run_dir"] == "/out/subject_s01"
    # config path is anchored on the Hydra output dir, artifacts on the run dir.
    assert row["config_path"] == "/out/.hydra/config.yaml"
    assert row["final_metrics_path"] == "/out/subject_s01/metrics/final_metrics.yaml"
    assert row["n_chans"] == 22 and row["n_outputs"] == 4
    assert row["dataset"] == "ds" and row["model"] == "eegnet"
    assert row["status"] == "success"


def test_build_master_row_held_out_subject_variant() -> None:
    row = build_master_row(
        _cfg(),
        {"test_accuracy": 0.4},
        _info(),
        run_id="heldout-s01__ts",
        run_dir=Path("/out/heldout-s01"),
        hydra_output_dir=Path("/out"),
        tb_dir=Path("/tb"),
        timestamp="ts",
        held_out_subject="heldout-s01",
    )

    assert row["held_out_subject"] == "heldout-s01"
    assert "subject" not in row


def test_order_fieldnames_leading_then_metrics_then_extra() -> None:
    rows = [{"subject": "s01", "test_accuracy": 0.5, "split_strategy": "x", "zzz_extra": 1}]

    fieldnames = order_fieldnames(rows, leading=["subject"])

    assert fieldnames[0] == "subject"
    assert fieldnames.index("split_strategy") < fieldnames.index("test_accuracy")
    assert fieldnames[-1] == "zzz_extra"
