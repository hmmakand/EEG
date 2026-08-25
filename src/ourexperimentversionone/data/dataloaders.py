"""PyTorch DataLoader factories for Liu2024 evaluation protocols."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset

from src.datautils.MoabbLiu2024 import Liu2024TorchDataset

from .config import DataLoaderConfig
from .splitters import DataSplit, group_kfold_splits, loso_split, within_subject_splits


@dataclass(frozen=True)
class DataLoaderBundle:
    """DataLoaders and the exact split from which they were constructed."""

    train: DataLoader
    validation: DataLoader | None
    test: DataLoader
    split: DataSplit


def _loader(
    dataset: Liu2024TorchDataset,
    indices,
    config: DataLoaderConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        Subset(dataset, list(map(int, indices))),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        generator=generator if shuffle else None,
    )


def _bundle(
    dataset: Liu2024TorchDataset,
    split: DataSplit,
    config: DataLoaderConfig,
) -> DataLoaderBundle:
    return DataLoaderBundle(
        train=_loader(dataset, split.train_indices, config, shuffle=True),
        validation=(
            _loader(dataset, split.validation_indices, config, shuffle=False)
            if len(split.validation_indices)
            else None
        ),
        test=_loader(dataset, split.test_indices, config, shuffle=False),
        split=split,
    )


def create_group_kfold_dataloaders(
    *,
    fold: int,
    config: DataLoaderConfig | None = None,
    dataset: Liu2024TorchDataset | None = None,
    n_splits: int = 5,
    validation_subjects: int = 8,
) -> DataLoaderBundle:
    """Create one subject-grouped development fold."""

    config = config or DataLoaderConfig()
    dataset = dataset or Liu2024TorchDataset()
    splits = group_kfold_splits(
        dataset.subject_ids,
        n_splits=n_splits,
        validation_subjects=validation_subjects,
        seed=config.seed,
    )
    if not 0 <= fold < len(splits):
        raise ValueError(f"fold must be between 0 and {len(splits) - 1}")
    return _bundle(dataset, splits[fold], config)


def create_loso_dataloaders(
    *,
    test_subject_id: int,
    config: DataLoaderConfig | None = None,
    dataset: Liu2024TorchDataset | None = None,
) -> DataLoaderBundle:
    """Create a leave-one-subject-out evaluation fold."""

    config = config or DataLoaderConfig()
    dataset = dataset or Liu2024TorchDataset()
    split = loso_split(
        dataset.subject_ids,
        test_subject_id=test_subject_id,
    )
    return _bundle(dataset, split, config)


def create_within_subject_dataloaders(
    *,
    subject_id: int,
    fold: int,
    config: DataLoaderConfig | None = None,
    dataset: Liu2024TorchDataset | None = None,
    n_splits: int = 5,
) -> DataLoaderBundle:
    """Create one personalized within-subject evaluation fold."""

    config = config or DataLoaderConfig()
    dataset = dataset or Liu2024TorchDataset()
    splits = within_subject_splits(
        dataset.subject_ids,
        dataset.y,
        subject_id=subject_id,
        n_splits=n_splits,
        seed=config.seed,
    )
    if not 0 <= fold < len(splits):
        raise ValueError(f"fold must be between 0 and {len(splits) - 1}")
    return _bundle(dataset, splits[fold], config)
