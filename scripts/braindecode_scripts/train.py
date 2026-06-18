from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_bci.data.datasets import build_datasets
from eeg_bci.models.factory import build_model
from eeg_bci.braindecode_training.trainer import train_model
from eeg_bci.utils.seed import seed_everything


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))

    device = _resolve_device(str(cfg.device))
    train_set, test_set, dataset_info = build_datasets(
        cfg.dataset,
        cfg.preprocessing,
        int(cfg.seed),
    )
    model = build_model(cfg.model, dataset_info)

    metrics = train_model(
        model,
        train_set,
        test_set,
        n_outputs=int(dataset_info.n_outputs),
        device=device,
        max_epochs=int(cfg.training.max_epochs),
        batch_size=int(cfg.training.batch_size),
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        num_workers=int(cfg.training.num_workers),
        output_dir=Path(cfg.output_dir),
        checkpoint_name=str(cfg.training.checkpoint_name),
        validation_enabled=bool(cfg.training.validation.enabled),
        validation_size=float(cfg.training.validation.valid_size),
        validation_shuffle=bool(cfg.training.validation.shuffle),
        seed=int(cfg.seed),
    )
    print("final_metrics:")
    print(OmegaConf.to_yaml(metrics))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


if __name__ == "__main__":
    main()
