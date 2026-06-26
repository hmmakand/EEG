from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


def write_run_text(
    log_dir: Path,
    *,
    cfg: DictConfig,
    summary: dict[str, Any],
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("run/summary", _markdown_dict(summary), global_step=0)
    writer.add_text("config/resolved", f"```yaml\n{OmegaConf.to_yaml(cfg)}\n```", global_step=0)
    for section in ("dataset", "model", "training", "preprocessing"):
        if section in cfg:
            writer.add_text(
                f"config/{section}",
                f"```yaml\n{OmegaConf.to_yaml(cfg[section])}\n```",
                global_step=0,
            )
    writer.flush()
    writer.close()


def write_history_scalars(log_dir: Path, classifier: Any) -> None:
    history = getattr(classifier, "history", None)
    if history is None:
        return
    writer = SummaryWriter(log_dir=str(log_dir))
    for row in history:
        if not isinstance(row, dict):
            continue
        step = int(row.get("epoch", 0))
        for source, target in {
            "train_loss": "train/loss",
            "train_accuracy": "train/accuracy",
            "valid_loss": "valid/loss",
            "valid_accuracy": "valid/accuracy",
            "event_lr": "learning_rate",
        }.items():
            value = row.get(source)
            if value is not None:
                writer.add_scalar(target, float(value), global_step=step)
    writer.flush()
    writer.close()


def write_final_scalars(log_dir: Path, metrics: dict[str, Any]) -> None:
    writer = SummaryWriter(log_dir=str(log_dir))
    written_tags: set[str] = set()

    legacy_mapping = {
        "test_acc": "test/accuracy",
        "cv_acc_mean": "cv/accuracy_mean",
        "cv_acc_std": "cv/accuracy_std",
        "best_score": "grid/best_score",
    }
    direct_mapping = {
        "best_score": "grid/best_score",
        "best_score_std": "grid/best_score_std",
        "best_train_score": "grid/best_train_score",
        "best_train_score_std": "grid/best_train_score_std",
    }

    for source, target in {**legacy_mapping, **direct_mapping}.items():
        scalar = _as_scalar_float(metrics.get(source))
        if scalar is not None and target not in written_tags:
            writer.add_scalar(target, scalar, global_step=0)
            written_tags.add(target)

    for key, value in metrics.items():
        scalar = _as_scalar_float(value)
        if scalar is None:
            continue
        tag = _metric_tag(key)
        if tag is None or tag in written_tags:
            continue
        writer.add_scalar(tag, scalar, global_step=0)
        written_tags.add(tag)

    writer.flush()
    writer.close()


def write_confusion_matrix(
    log_dir: Path,
    cm: list[list[int]] | np.ndarray,
    *,
    class_names: list[str] | None = None,
    tag: str = "test/confusion_matrix",
    global_step: int = 0,
) -> None:
    """Log a confusion matrix as a heatmap image in TensorBoard."""
    cm_array = np.asarray(cm)
    if cm_array.ndim != 2 or cm_array.shape[0] != cm_array.shape[1]:
        raise ValueError(f"confusion matrix must be square; got shape {cm_array.shape}.")

    labels = class_names or [f"class_{index}" for index in range(cm_array.shape[0])]
    if len(labels) != cm_array.shape[0]:
        raise ValueError(
            f"class_names length ({len(labels)}) must match confusion-matrix size "
            f"({cm_array.shape[0]})."
        )

    figure = Figure(figsize=(max(6, len(labels)), max(5, len(labels))), dpi=100)
    axes = figure.add_subplot(111)
    image = axes.imshow(cm_array, cmap="Blues")
    axes.set_xticks(np.arange(len(labels)))
    axes.set_yticks(np.arange(len(labels)))
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.set_yticklabels(labels)
    axes.set_ylabel("True")
    axes.set_xlabel("Predicted")
    figure.colorbar(image, ax=axes)

    threshold = float(cm_array.max()) / 2.0 if cm_array.size else 0.0
    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            axes.text(
                col_index,
                row_index,
                str(cm_array[row_index, col_index]),
                ha="center",
                va="center",
                color="white" if cm_array[row_index, col_index] > threshold else "black",
            )

    figure.tight_layout()
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    width, height = figure.canvas.get_width_height()
    image_array = np.asarray(canvas.buffer_rgba()).reshape(height, width, 4)

    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_image(tag, image_array, dataformats="HWC", global_step=global_step)
    writer.flush()
    writer.close()


def _metric_tag(key: str) -> str | None:
    if key.startswith("test_"):
        return f"test/{key[5:]}"
    if key.startswith("cv_accuracy_"):
        return f"cv/accuracy_{key[len('cv_accuracy_'):]}"
    return None


def _as_scalar_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return float(value)


def _markdown_dict(payload: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in payload.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)
