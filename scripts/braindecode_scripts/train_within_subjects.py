from __future__ import annotations

import csv
from collections.abc import Sequence
import sys
from pathlib import Path

import hydra
import torch
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
    print(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))

    device = _resolve_device(str(cfg.device))
    output_dir = Path(cfg.output_dir)
    subject_ids = _subject_ids(cfg.dataset)
    results: list[dict[str, str | float | int]] = []

    for subject_id in subject_ids:
        print(f"subject={subject_id}: starting")
        subject_cfg = _copy_dataset_cfg(cfg.dataset)
        if subject_id != "all":
            subject_cfg.subject_ids = [int(subject_id)]

        split_plan, dataset_info = build_dataset_split(
            subject_cfg,
            cfg.preprocessing,
            int(cfg.seed),
        )
        model = build_model(cfg.model, dataset_info)
        subject_output_dir = output_dir / f"subject_{subject_id}"
        metrics = train_from_split_plan(
            model,
            split_plan,
            n_outputs=int(dataset_info.n_outputs),
            device=device,
            training_cfg=cfg.training,
            output_dir=subject_output_dir,
        )
        row = {"subject": subject_id, **metrics}
        results.append(row)
        print(f"subject={subject_id}: done")
        print(OmegaConf.to_yaml(row))

    _write_results(output_dir / "within_subject_results.csv", results)
    print(f"within_subject_results: {output_dir / 'within_subject_results.csv'}")


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
        "subject",
        "split_strategy",
        "train_loss",
        "train_acc",
        "valid_loss",
        "valid_acc",
        "cv_acc_mean",
        "cv_acc_std",
        "best_score",
        "best_score_std",
        "best_train_score",
        "best_train_score_std",
        "best_params",
        "test_acc",
    ]
    present = {key for row in results for key in row}
    ordered = [fieldname for fieldname in preferred if fieldname in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


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
