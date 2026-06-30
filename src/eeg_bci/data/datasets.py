from __future__ import annotations

from collections.abc import Callable

from omegaconf import DictConfig
from torch.utils.data import Dataset

from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.moabb import build_moabb_dataset
from eeg_bci.data.splitting import SplitPlan, make_protocol_split
from eeg_bci.data.synthetic import build_synthetic_dataset
from eeg_bci.data.types import DatasetInfo

DatasetBuilder = Callable[[DictConfig, DictConfig], tuple[BraindecodeLikeDataset, DatasetInfo]]

DATASET_BUILDERS: dict[str, DatasetBuilder] = {
    "synthetic": build_synthetic_dataset,
    "moabb": build_moabb_dataset,
}


def build_dataset_split(
    dataset_cfg: DictConfig,
    preprocessing_cfg: DictConfig,
    seed: int,
) -> tuple[SplitPlan, DatasetInfo]:
    dataset, info = build_dataset(dataset_cfg, preprocessing_cfg)
    split_cfg = dataset_cfg.get("split", None)
    return make_protocol_split(dataset, split_cfg, seed=seed), info


def build_datasets(
    dataset_cfg: DictConfig,
    preprocessing_cfg: DictConfig,
    seed: int,
) -> tuple[Dataset, Dataset, DatasetInfo]:
    split_plan, info = build_dataset_split(dataset_cfg, preprocessing_cfg, seed)
    return split_plan.train_pool, split_plan.test_set, info


def build_dataset(
    dataset_cfg: DictConfig,
    preprocessing_cfg: DictConfig,
) -> tuple[BraindecodeLikeDataset, DatasetInfo]:
    builder = get_dataset_builder(str(dataset_cfg.name))
    return builder(dataset_cfg, preprocessing_cfg)


def get_dataset_builder(name: str) -> DatasetBuilder:
    try:
        return DATASET_BUILDERS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_BUILDERS))
        raise ValueError(
            f"Unsupported dataset {name}. Available datasets: {available}."
        ) from exc
