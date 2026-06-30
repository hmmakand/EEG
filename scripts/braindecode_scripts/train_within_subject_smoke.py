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
from eeg_bci.tracking.artifacts import (
    prepare_run_dirs,
    save_dataset_info,
    save_final_metrics,
    save_run_metadata,
)
from eeg_bci.tracking.logging import configure_logging
from eeg_bci.tracking.naming import (
    class_names_from_mapping,
    dataset_label,
    model_label,
    subject_scope,
    tensorboard_dir,
)
from eeg_bci.tracking.results import append_master_result, master_result_path
from eeg_bci.tracking.tensorboard import write_run_text
from eeg_bci.utils.seed import seed_everything

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    prepare_run_dirs(output_dir)
    configure_logging(output_dir / "logs")

    if not _is_within_subject_smoke_experiment():
        logger.error(
            "train_within_subject_smoke.py is only for experiments ending with 'within_subject_smoke'."
        )
        return

    logger.info("resolved_config:\n%s", OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))

    scope = subject_scope(cfg.dataset.subject_ids)
    run_id = f"{scope}__{output_dir.name}"
    tb_dir = tensorboard_dir(cfg, output_dir.name, scope)
    save_run_metadata(output_dir, cfg=cfg, run_id=run_id, tensorboard_dir=tb_dir)

    device = _resolve_device(str(cfg.device))
    split_plan, dataset_info = build_dataset_split(
        cfg.dataset,
        cfg.preprocessing,
        int(cfg.seed),
    )
    model = build_model(cfg.model, dataset_info)

    metrics = train_from_split_plan(
        model,
        split_plan,
        n_outputs=int(dataset_info.n_outputs),
        device=device,
        training_cfg=cfg.training,
        output_dir=output_dir,
        tensorboard_dir=tb_dir,
        class_names=class_names_from_mapping(cfg.dataset),
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
        master_result_path(
            Path(get_original_cwd()),
            str(cfg.experiment_name),
            dataset_label(cfg.dataset),
            str(cfg.dataset.split.method),
        ),
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
            "final_metrics_path": str(output_dir / "metrics" / "final_metrics.yaml"),
            "dataset_info_path": str(output_dir / "metrics" / "dataset_info.yaml"),
            "status": "success",
        },
    )
    save_run_metadata(output_dir, cfg=cfg, run_id=run_id, tensorboard_dir=tb_dir, status="success")
    logger.info("final_metrics:\n%s", OmegaConf.to_yaml(metrics))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _is_within_subject_smoke_experiment() -> bool:
    experiment = HydraConfig.get().runtime.choices.get("experiment")
    if experiment is None:
        return False
    return str(experiment).endswith("within_subject_smoke")


if __name__ == "__main__":
    if not any(arg.startswith("experiment=") for arg in sys.argv[1:]):
        sys.argv.append("experiment=within_subject_smoke")
    main()
