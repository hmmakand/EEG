from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_bci.data.datasets import build_dataset_split
from eeg_bci.models.factory import build_model
from eeg_bci.braindecode_training.trainer import train_from_split_plan
from eeg_bci.utils.seed import seed_everything


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not _is_smoke_experiment():
        print(
            "train_single_run.py is only for the smoke experiment.\n"
            "Run: python scripts/braindecode_scripts/train_single_run.py experiment=smoke\n"
            "For the full within-subject run, use: python scripts/braindecode_scripts/train_eval_within_subjects.py experiment=full"
        )
        return
    print(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))

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
        output_dir=Path(cfg.output_dir),
    )
    print("final_metrics:")
    print(OmegaConf.to_yaml(metrics))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _is_smoke_experiment() -> bool:
    experiment = HydraConfig.get().runtime.choices.get("experiment")
    return experiment == "smoke"


if __name__ == "__main__":
    if not any(arg.startswith("experiment=") for arg in sys.argv[1:]):
        sys.argv.append("experiment=smoke")
    main()
