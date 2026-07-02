"""Shared helpers for the ``train_*`` entry-point scripts.

These consolidate boilerplate that was previously copy-pasted across every
script: device resolution and assembly of the master-result CSV row. Keeping it
here (in the package, which the scripts add to ``sys.path``) means adding a
column or metric is a single-site change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from eeg_bci.data.types import DatasetInfo
from eeg_bci.tracking.naming import dataset_label, model_label


def resolve_device(device_name: str) -> torch.device:
    """Resolve a configured device string, mapping ``"auto"`` to CUDA/CPU."""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_master_row(
    cfg: DictConfig,
    metrics: dict[str, Any],
    dataset_info: DatasetInfo,
    *,
    run_id: str,
    run_dir: Path,
    hydra_output_dir: Path,
    tb_dir: Path,
    timestamp: str,
    subject: Any | None = None,
    held_out_subject: Any | None = None,
    status: str = "success",
) -> dict[str, Any]:
    """Assemble one master-result CSV row shared by all training scripts.

    ``run_dir`` is the per-run directory (subject/fold or top-level) used for
    the metrics/dataset-info artifact paths; ``hydra_output_dir`` is Hydra's
    runtime output directory used for the resolved-config path (identical to
    ``run_dir`` for single-run scripts).
    """

    row: dict[str, Any] = {
        **metrics,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "tensorboard_dir": str(tb_dir),
        "timestamp": timestamp,
        "experiment": str(cfg.experiment_name),
        "dataset": dataset_label(cfg.dataset),
        "model": model_label(cfg.model),
        "seed": int(cfg.seed),
        "n_chans": dataset_info.n_chans,
        "n_outputs": dataset_info.n_outputs,
        "n_times": dataset_info.n_times,
        "sfreq": dataset_info.sfreq,
        "config_path": str(hydra_output_dir / ".hydra" / "config.yaml"),
        "final_metrics_path": str(run_dir / "metrics" / "final_metrics.yaml"),
        "dataset_info_path": str(run_dir / "metrics" / "dataset_info.yaml"),
        "status": status,
    }
    if subject is not None:
        row["subject"] = subject
    if held_out_subject is not None:
        row["held_out_subject"] = held_out_subject
    return row
