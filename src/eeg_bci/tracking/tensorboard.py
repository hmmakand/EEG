from __future__ import annotations

from pathlib import Path
from typing import Any

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
    for source, target in {
        "test_acc": "test/accuracy",
        "cv_acc_mean": "cv/accuracy_mean",
        "cv_acc_std": "cv/accuracy_std",
        "best_score": "grid/best_score",
    }.items():
        if source in metrics:
            writer.add_scalar(target, float(metrics[source]), global_step=0)
    writer.flush()
    writer.close()


def _markdown_dict(payload: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in payload.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)

