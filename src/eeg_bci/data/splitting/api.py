"""Backward-compatible high-level split helpers."""

from __future__ import annotations

from omegaconf import DictConfig
from torch.utils.data import Dataset

from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.splitting.config import configured_strategy
from eeg_bci.data.splitting.plans import make_braindecode_protocol_split
from eeg_bci.data.splitting.sources import (
    split_chronological_train_test,
    split_random_train_test,
)
from eeg_bci.data.splitting.strategies import (
    CHRONOLOGICAL_CROSS_VALIDATION_TEST,
    CHRONOLOGICAL_GRID_SEARCH_TEST,
    CHRONOLOGICAL_TRAIN_TEST,
    CHRONOLOGICAL_TRAIN_VALID_TEST,
    SESSION_SPLIT_STRATEGIES,
)


def split_train_test(
    dataset: BraindecodeLikeDataset,
    dataset_cfg: DictConfig,
    *,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into training and final test sets."""

    split_cfg = dataset_cfg.get("split", None)
    strategy = configured_strategy(split_cfg)

    if strategy in SESSION_SPLIT_STRATEGIES or strategy == "description":
        split_plan = make_braindecode_protocol_split(dataset, split_cfg, seed=seed)
        return split_plan.train_pool, split_plan.test_set

    if strategy == "random":
        test_size = (
            float(split_cfg.get("test_size", dataset_cfg.get("test_size", 0.2)))
            if split_cfg is not None
            else float(dataset_cfg.get("test_size", 0.2))
        )
        return split_random_train_test(dataset, test_size=test_size, seed=seed)

    if strategy == CHRONOLOGICAL_TRAIN_TEST:
        test_size = float(split_cfg.get("test_size", 0.2))
        stratify = bool(split_cfg.get("stratify", True))
        return split_chronological_train_test(
            dataset, test_size=test_size, stratify=stratify, seed=seed
        )

    if strategy in {
        CHRONOLOGICAL_TRAIN_VALID_TEST,
        CHRONOLOGICAL_CROSS_VALIDATION_TEST,
        CHRONOLOGICAL_GRID_SEARCH_TEST,
    }:
        raise ValueError(
            f"Split strategy {strategy} returns a SplitPlan and must be built with "
            "build_dataset_split(), not split_train_test()."
        )

    raise ValueError(f"Unsupported split strategy {strategy}.")


def split_train_eval(
    dataset: BraindecodeLikeDataset,
    dataset_cfg: DictConfig,
    *,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Backward-compatible alias for split_train_test."""

    return split_train_test(dataset, dataset_cfg, seed=seed)
