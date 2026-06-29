from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Any, Protocol, cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_bci.braindecode_training.trainer import train_from_split_plan
from eeg_bci.data.datasets import build_dataset
from eeg_bci.data.splitting import make_leave_one_subject_out_folds
from eeg_bci.models.factory import build_model
from eeg_bci.tracking.artifacts import prepare_run_dirs, save_dataset_info, save_final_metrics, save_run_metadata
from eeg_bci.tracking.logging import configure_logging
from eeg_bci.tracking.metrics import summarize_scalar_metrics
from eeg_bci.tracking.naming import (
    class_names_from_mapping,
    dataset_label,
    held_out_label,
    model_label,
    tensorboard_dir,
)
from eeg_bci.tracking.results import append_master_result, master_result_path
from eeg_bci.tracking.tensorboard import write_run_text
from eeg_bci.utils.seed import seed_everything

logger = logging.getLogger(__name__)


class SizedDataset(Protocol):
    def __len__(self) -> int: ...


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    prepare_run_dirs(output_dir)
    configure_logging(output_dir / "logs")

    if not _is_loso_experiment():
        logger.error("train_loso.py is only for experiments ending with 'loso'.")
        return

    logger.info("resolved_config:\n%s", OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))

    parent_run_id = f"all_subjects__{output_dir.name}"
    save_run_metadata(
        output_dir,
        cfg=cfg,
        run_id=parent_run_id,
        tensorboard_dir=tensorboard_dir(cfg, output_dir.name),
    )

    device = _resolve_device(str(cfg.device))
    dataset, dataset_info = build_dataset(cfg.dataset, cfg.preprocessing)
    validation_cfg = cfg.training.validation
    folds = make_leave_one_subject_out_folds(
        dataset,
        cfg.dataset.split,
        seed=int(cfg.seed),
        validation_enabled=bool(validation_cfg.enabled),
        validation_size=float(validation_cfg.valid_size),
        validation_shuffle=bool(validation_cfg.shuffle),
    )

    results: list[dict[str, Any]] = []
    for fold in folds:
        held_out = fold.held_out_subject
        label = held_out_label(held_out)
        logger.info("held_out_subject=%s: starting", held_out)
        model = build_model(cfg.model, dataset_info)
        fold_output_dir = output_dir / label
        prepare_run_dirs(fold_output_dir)
        run_id = f"{label}__{output_dir.name}"
        tb_dir = tensorboard_dir(cfg, output_dir.name, label)
        split_plan = fold.split_plan
        metrics = train_from_split_plan(
            model,
            split_plan,
            n_outputs=int(dataset_info.n_outputs),
            device=device,
            training_cfg=cfg.training,
            output_dir=fold_output_dir,
            tensorboard_dir=tb_dir,
            class_names=class_names_from_mapping(cfg.dataset),
        )
        row = {
            "held_out_subject": held_out,
            "train_subjects": " ".join(str(subject) for subject in fold.train_subjects),
            "n_train_windows": len(cast(SizedDataset, fold.train_set)),
            "n_valid_windows": _dataset_len(fold.valid_set),
            "n_test_windows": len(cast(SizedDataset, fold.test_set)),
            **metrics,
        }
        results.append(row)
        save_final_metrics(fold_output_dir, metrics)
        save_dataset_info(
            fold_output_dir,
            dataset_info,
            extra={
                "dataset": dataset_label(cfg.dataset),
                "held_out_subject": held_out,
                "train_subjects": list(fold.train_subjects),
                "split_strategy": metrics.get("split_strategy"),
                "n_train_windows": row["n_train_windows"],
                "n_valid_windows": row["n_valid_windows"],
                "n_test_windows": row["n_test_windows"],
            },
        )
        write_run_text(
            tb_dir,
            cfg=cfg,
            summary={
                "run_id": run_id,
                "experiment": str(cfg.experiment_name),
                "dataset": dataset_label(cfg.dataset),
                "model": model_label(cfg.model),
                "held_out_subject": label,
                "seed": int(cfg.seed),
                "split_strategy": metrics.get("split_strategy"),
                "device": str(device),
                "run_dir": fold_output_dir,
            },
        )
        append_master_result(
            master_result_path(
                Path(get_original_cwd()),
                str(cfg.experiment_name),
                dataset_label(cfg.dataset),
            ),
            {
                **metrics,
                "run_id": run_id,
                "run_dir": str(fold_output_dir),
                "tensorboard_dir": str(tb_dir),
                "timestamp": output_dir.name,
                "experiment": str(cfg.experiment_name),
                "dataset": dataset_label(cfg.dataset),
                "model": model_label(cfg.model),
                "held_out_subject": label,
                "seed": int(cfg.seed),
                "n_chans": dataset_info.n_chans,
                "n_outputs": dataset_info.n_outputs,
                "n_times": dataset_info.n_times,
                "sfreq": dataset_info.sfreq,
                "config_path": str(output_dir / ".hydra" / "config.yaml"),
                "final_metrics_path": str(fold_output_dir / "metrics" / "final_metrics.yaml"),
                "dataset_info_path": str(fold_output_dir / "metrics" / "dataset_info.yaml"),
                "status": "success",
            },
        )
        save_run_metadata(fold_output_dir, cfg=cfg, run_id=run_id, tensorboard_dir=tb_dir, status="success")
        logger.info("held_out_subject=%s: done", held_out)
        logger.info("fold_metrics:\n%s", OmegaConf.to_yaml(row))

    results_path = output_dir / "results" / "loso_results.csv"
    _write_results(results_path, results)
    summary = _summarize_results(results)
    save_final_metrics(output_dir, {f"loso_{key}": value for key, value in summary.items()})
    save_dataset_info(
        output_dir,
        dataset_info,
        extra={
            "dataset": dataset_label(cfg.dataset),
            "split_strategy": str(cfg.dataset.split.strategy),
        },
    )
    save_run_metadata(
        output_dir,
        cfg=cfg,
        run_id=parent_run_id,
        tensorboard_dir=tensorboard_dir(cfg, output_dir.name),
        status="success",
    )
    logger.info("loso_results=%s", results_path)
    for key, value in summary.items():
        logger.info("loso_%s=%.4f", key, value)


def _dataset_len(dataset: object | None) -> int:
    if dataset is None:
        return 0
    return len(cast(SizedDataset, dataset))


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, float]:
    return summarize_scalar_metrics(results)


def _result_fieldnames(results: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "held_out_subject",
        "train_subjects",
        "n_train_windows",
        "n_valid_windows",
        "n_test_windows",
        "split_strategy",
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
        "checkpoint_path",
        "history_path",
    ]
    present = {key for row in results for key in row}
    ordered = [fieldname for fieldname in preferred if fieldname in present]
    return ordered + sorted(present.difference(ordered))


def _write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _result_fieldnames(results)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _is_loso_experiment() -> bool:
    experiment = HydraConfig.get().runtime.choices.get("experiment")
    if experiment is None:
        return False
    return str(experiment).endswith("loso")


if __name__ == "__main__":
    if not any(arg.startswith("experiment=") for arg in sys.argv[1:]):
        sys.argv.append("experiment=bcic_iv_2a_loso")
    main()
