from __future__ import annotations

import csv
from collections.abc import Sequence
import logging
import sys
from pathlib import Path

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
from eeg_bci.data.datasets import build_dataset_split
from eeg_bci.models.factory import build_model
from eeg_bci.tracking.artifacts import prepare_run_dirs, save_dataset_info, save_final_metrics, save_run_metadata
from eeg_bci.tracking.logging import configure_logging
from eeg_bci.tracking.naming import dataset_label, model_label, subject_label, tensorboard_dir
from eeg_bci.tracking.results import append_master_result
from eeg_bci.tracking.tensorboard import write_run_text
from eeg_bci.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    prepare_run_dirs(output_dir)
    configure_logging(output_dir / "logs")
    logger.info("resolved_config:\n%s", OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))

    device = _resolve_device(str(cfg.device))
    subject_ids = _subject_ids(cfg.dataset)
    results: list[dict[str, str | float | int]] = []
    parent_run_id = f"all_subjects__{output_dir.name}"
    save_run_metadata(output_dir, cfg=cfg, run_id=parent_run_id, tensorboard_dir=tensorboard_dir(cfg, parent_run_id))

    for subject_id in subject_ids:
        label = subject_label(subject_id)
        logger.info("subject=%s: starting", subject_id)
        subject_cfg = _copy_dataset_cfg(cfg.dataset)
        if subject_id != "all":
            subject_cfg.subject_ids = [int(subject_id)]

        split_plan, dataset_info = build_dataset_split(subject_cfg, cfg.preprocessing, int(cfg.seed))
        model = build_model(cfg.model, dataset_info)
        subject_output_dir = output_dir / f"subject_{label}"
        prepare_run_dirs(subject_output_dir)
        run_id = f"{label}__{output_dir.name}"
        tb_dir = tensorboard_dir(cfg, run_id)
        metrics = train_from_split_plan(
            model,
            split_plan,
            n_outputs=int(dataset_info.n_outputs),
            device=device,
            training_cfg=cfg.training,
            output_dir=subject_output_dir,
            tensorboard_dir=tb_dir,
        )
        row = {"subject": subject_id, **metrics}
        results.append(row)
        save_final_metrics(subject_output_dir, metrics)
        save_dataset_info(
            subject_output_dir,
            dataset_info,
            extra={
                "dataset": dataset_label(cfg.dataset),
                "subject": subject_id,
                "split_strategy": metrics.get("split_strategy"),
                "n_train_windows": metrics.get("n_train_windows"),
                "n_valid_windows": metrics.get("n_valid_windows"),
                "n_test_windows": metrics.get("n_test_windows"),
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
                "subject": label,
                "seed": int(cfg.seed),
                "split_strategy": metrics.get("split_strategy"),
                "device": str(device),
                "run_dir": subject_output_dir,
            },
        )
        append_master_result(
            Path(get_original_cwd()) / "outputs" / "results_master.csv",
            {
                **metrics,
                "run_id": run_id,
                "run_dir": str(subject_output_dir),
                "tensorboard_dir": str(tb_dir),
                "timestamp": output_dir.name,
                "experiment": str(cfg.experiment_name),
                "dataset": dataset_label(cfg.dataset),
                "model": model_label(cfg.model),
                "subject": label,
                "seed": int(cfg.seed),
                "n_chans": dataset_info.n_chans,
                "n_outputs": dataset_info.n_outputs,
                "n_times": dataset_info.n_times,
                "sfreq": dataset_info.sfreq,
                "config_path": str(output_dir / ".hydra" / "config.yaml"),
                "status": "success",
            },
        )
        save_run_metadata(subject_output_dir, cfg=cfg, run_id=run_id, tensorboard_dir=tb_dir, status="success")
        logger.info("subject=%s: done", subject_id)
        logger.info("subject_metrics:\n%s", OmegaConf.to_yaml(row))

    results_path = output_dir / "results" / "within_subject_results.csv"
    _write_results(results_path, results)
    save_run_metadata(output_dir, cfg=cfg, run_id=parent_run_id, tensorboard_dir=tensorboard_dir(cfg, parent_run_id), status="success")
    logger.info("within_subject_results=%s", results_path)


def _copy_dataset_cfg(dataset_cfg: DictConfig) -> DictConfig:
    copied = OmegaConf.create(OmegaConf.to_container(dataset_cfg, resolve=True))
    if not isinstance(copied, DictConfig):
        raise TypeError("Expected dataset config copy to be a DictConfig.")
    return copied


def _subject_ids(dataset_cfg: DictConfig) -> Sequence[int | str]:
    if "subject_ids" not in dataset_cfg:
        return ["all"]
    subject_ids = OmegaConf.to_container(dataset_cfg.subject_ids, resolve=True)
    if subject_ids is None:
        return ["all"]
    if isinstance(subject_ids, int):
        return [subject_ids]
    if not isinstance(subject_ids, list):
        raise TypeError("dataset.subject_ids must be an int, a list of ints, or null.")
    return [int(subject_id) for subject_id in subject_ids]


def _result_fieldnames(results: list[dict[str, str | float | int]]) -> list[str]:
    preferred = [
        "subject", "split_strategy", "n_train_windows", "n_valid_windows", "n_test_windows",
        "train_loss", "train_acc", "valid_loss", "valid_acc", "cv_acc_mean", "cv_acc_std",
        "best_score", "best_score_std", "best_train_score", "best_train_score_std",
        "best_params", "test_acc", "checkpoint_path", "history_path",
    ]
    present = {key for row in results for key in row}
    ordered = [fieldname for fieldname in preferred if fieldname in present]
    return ordered + sorted(present.difference(ordered))


def _write_results(path: Path, results: list[dict[str, str | float | int]]) -> None:
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


if __name__ == "__main__":
    main()
