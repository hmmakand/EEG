from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import braindecode
import torch
from omegaconf import DictConfig, OmegaConf

from eeg_bci.data.types import DatasetInfo


MetricValue = float | int | str


def prepare_run_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "logs": output_dir / "logs",
        "metrics": output_dir / "metrics",
        "history": output_dir / "history",
        "checkpoints": output_dir / "checkpoints",
        "results": output_dir / "results",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def save_final_metrics(output_dir: Path, metrics: dict[str, MetricValue]) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _save_yaml(metrics_dir / "final_metrics.yaml", metrics)
    _save_json(metrics_dir / "final_metrics.json", metrics)


def save_dataset_info(
    output_dir: Path,
    dataset_info: DatasetInfo,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "n_chans": dataset_info.n_chans,
        "n_outputs": dataset_info.n_outputs,
        "n_times": dataset_info.n_times,
        "sfreq": dataset_info.sfreq,
    }
    if extra:
        payload.update(extra)
    _save_yaml(output_dir / "metrics" / "dataset_info.yaml", payload)


def save_run_metadata(
    output_dir: Path,
    *,
    cfg: DictConfig,
    run_id: str,
    tensorboard_dir: Path,
    status: str = "running",
) -> None:
    metadata = {
        "run_id": run_id,
        "run_dir": str(output_dir),
        "tensorboard_dir": str(tensorboard_dir),
        "experiment": _choice_or_none(cfg, "experiment"),
        "seed": int(cfg.seed),
        "device": str(cfg.device),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "braindecode_version": getattr(braindecode, "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_git_output(["git", "status", "--porcelain"])),
        "status": status,
    }
    _save_yaml(output_dir / "metrics" / "run_metadata.yaml", metadata)


def export_history(classifier: Any, output_dir: Path) -> Path:
    rows = _history_rows(classifier)
    path = output_dir / "history" / "history.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    fieldnames = _ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _history_rows(classifier: Any) -> list[dict[str, Any]]:
    history = getattr(classifier, "history", None)
    if history is None:
        return []
    rows: list[dict[str, Any]] = []
    for row in history:
        if isinstance(row, dict):
            rows.append(_jsonable_dict(row))
    return rows


def _ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "valid_loss",
        "valid_accuracy",
        "event_lr",
        "dur",
    ]
    present = {key for row in rows for key in row}
    return [key for key in preferred if key in present] + sorted(present.difference(preferred))


def _save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(_jsonable_dict(payload)), path)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable_dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _jsonable_dict(payload: dict[str, Any]) -> dict[str, Any]:
    skipped = {"batches"}
    return {str(key): _jsonable(value) for key, value in payload.items() if str(key) not in skipped}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return _jsonable_dict(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _choice_or_none(cfg: DictConfig, key: str) -> str | None:
    try:
        from hydra.core.hydra_config import HydraConfig

        value = HydraConfig.get().runtime.choices.get(key)
    except ValueError:
        value = None
    return None if value is None else str(value)


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""

