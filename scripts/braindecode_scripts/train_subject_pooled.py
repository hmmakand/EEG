from __future__ import annotations

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
from eeg_bci.tracking.naming import dataset_label, model_label, subject_scope, tensorboard_dir
from eeg_bci.tracking.results import append_master_result
from eeg_bci.tracking.tensorboard import write_run_text
from eeg_bci.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    prepare_run_dirs(output_dir)
    configure_logging(output_dir / "logs")

    if not _is_subject_pooled_experiment():
        logger.error("train_subject_pooled.py is only for experiment=subject_pooled.")
        return

    logger.info("resolved_config:\n%s", OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))

    run_id = f"{subject_scope(cfg.dataset.subject_ids)}__{output_dir.name}"
    tb_dir = tensorboard_dir(cfg, run_id)
    save_run_metadata(output_dir, cfg=cfg, run_id=run_id, tensorboard_dir=tb_dir)

    device = _resolve_device(str(cfg.device))
    split_plan, dataset_info = build_dataset_split(cfg.dataset, cfg.preprocessing, int(cfg.seed))
    model = build_model(cfg.model, dataset_info)

    metrics = train_from_split_plan(
        model,
        split_plan,
        n_outputs=int(dataset_info.n_outputs),
        device=device,
        training_cfg=cfg.training,
        output_dir=output_dir,
        tensorboard_dir=tb_dir,
    )
    save_final_metrics(output_dir, metrics)
    save_dataset_info(
        output_dir,
        dataset_info,
        extra={
            "dataset": dataset_label(cfg.dataset),
            "subject_ids": OmegaConf.to_container(cfg.dataset.subject_ids, resolve=True),
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
            "subject_scope": subject_scope(cfg.dataset.subject_ids),
            "seed": int(cfg.seed),
            "split_strategy": metrics.get("split_strategy"),
            "device": str(device),
            "run_dir": output_dir,
        },
    )
    append_master_result(
        Path(get_original_cwd()) / "outputs" / "results_master.csv",
        {
            **metrics,
            "run_id": run_id,
            "run_dir": str(output_dir),
            "tensorboard_dir": str(tb_dir),
            "timestamp": output_dir.name,
            "experiment": str(cfg.experiment_name),
            "dataset": dataset_label(cfg.dataset),
            "model": model_label(cfg.model),
            "subject": subject_scope(cfg.dataset.subject_ids),
            "seed": int(cfg.seed),
            "n_chans": dataset_info.n_chans,
            "n_outputs": dataset_info.n_outputs,
            "n_times": dataset_info.n_times,
            "sfreq": dataset_info.sfreq,
            "config_path": str(output_dir / ".hydra" / "config.yaml"),
            "status": "success",
        },
    )
    save_run_metadata(output_dir, cfg=cfg, run_id=run_id, tensorboard_dir=tb_dir, status="success")
    logger.info("final_metrics:\n%s", OmegaConf.to_yaml(metrics))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _is_subject_pooled_experiment() -> bool:
    experiment = HydraConfig.get().runtime.choices.get("experiment")
    return experiment == "subject_pooled"


if __name__ == "__main__":
    if not any(arg.startswith("experiment=") for arg in sys.argv[1:]):
        sys.argv.append("experiment=subject_pooled")
    main()
