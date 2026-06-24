from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Protocol, cast

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eeg_bci.braindecode_training.trainer import train_from_split_plan
from eeg_bci.data.datasets import build_dataset
from eeg_bci.data.splitting import LEAVE_ONE_SUBJECT_OUT, SplitPlan
from eeg_bci.data.splitting import make_leave_one_subject_out_folds
from eeg_bci.models.factory import build_model
from eeg_bci.utils.seed import seed_everything


class SizedDataset(Protocol):
    def __len__(self) -> int: ...


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not _is_loso_experiment():
        print(
            "train_loso.py is only for the leave-one-subject-out experiment.\n"
            "Run: python scripts/braindecode_scripts/train_loso.py experiment=loso"
        )
        return

    print(OmegaConf.to_yaml(cfg))
    seed_everything(int(cfg.seed))

    device = _resolve_device(str(cfg.device))
    output_dir = Path(cfg.output_dir)
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

    results: list[dict[str, str | float | int]] = []
    for fold in folds:
        held_out = fold.held_out_subject
        print(f"held_out_subject={held_out}: starting")
        model = build_model(cfg.model, dataset_info)
        fold_output_dir = output_dir / f"held_out_subject_{held_out}"
        split_plan = SplitPlan(
            split_strategy=LEAVE_ONE_SUBJECT_OUT,
            train_pool=fold.train_set,
            train_set=fold.train_set,
            valid_set=fold.valid_set,
            test_set=fold.test_set,
            resampler=None,
        )
        metrics = train_from_split_plan(
            model,
            split_plan,
            n_outputs=int(dataset_info.n_outputs),
            device=device,
            training_cfg=cfg.training,
            output_dir=fold_output_dir,
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
        print(f"held_out_subject={held_out}: done")
        print(OmegaConf.to_yaml(row))

    results_path = output_dir / "loso_results.csv"
    _write_results(results_path, results)
    summary = _summarize_results(results)
    print(f"loso_results: {results_path}")
    test_acc_mean = summary["test_acc_mean"]
    test_acc_std = summary["test_acc_std"]
    test_acc_weighted = summary["test_acc_weighted"]
    print(
        f"loso_test_acc_mean={test_acc_mean:.4f} "
        f"loso_test_acc_std={test_acc_std:.4f} "
        f"loso_test_acc_weighted={test_acc_weighted:.4f}"
    )


def _dataset_len(dataset: object | None) -> int:
    if dataset is None:
        return 0
    return len(cast(SizedDataset, dataset))


def _summarize_results(
    results: list[dict[str, str | float | int]],
) -> dict[str, float]:
    test_accs = [float(row["test_acc"]) for row in results]
    test_counts = [int(row["n_test_windows"]) for row in results]
    if not test_accs:
        return {
            "test_acc_mean": 0.0,
            "test_acc_std": 0.0,
            "test_acc_weighted": 0.0,
        }

    mean = sum(test_accs) / len(test_accs)
    variance = sum((score - mean) ** 2 for score in test_accs) / len(test_accs)
    total_count = sum(test_counts)
    weighted = (
        sum(score * count for score, count in zip(test_accs, test_counts)) / total_count
        if total_count > 0
        else 0.0
    )
    return {
        "test_acc_mean": mean,
        "test_acc_std": variance ** 0.5,
        "test_acc_weighted": weighted,
    }


def _result_fieldnames(results: list[dict[str, str | float | int]]) -> list[str]:
    preferred = [
        "held_out_subject",
        "train_subjects",
        "n_train_windows",
        "n_valid_windows",
        "n_test_windows",
        "split_strategy",
        "train_loss",
        "train_acc",
        "valid_loss",
        "valid_acc",
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


def _is_loso_experiment() -> bool:
    experiment = HydraConfig.get().runtime.choices.get("experiment")
    return experiment == "loso"


if __name__ == "__main__":
    if not any(arg.startswith("experiment=") for arg in sys.argv[1:]):
        sys.argv.append("experiment=loso")
    main()
