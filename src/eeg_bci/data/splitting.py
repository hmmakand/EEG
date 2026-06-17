"""Dataset splitting utilities.

The project keeps the final test split separate from optional validation:

- Synthetic smoke-test datasets use a deterministic random train/test split.
- Real EEG datasets can use Braindecode description-based train/test splitting,
  such as BCI IV 2a where `session=0train` is training data and
  `session=1test` is the final test data.
- Optional validation is split only from the training set, never from the test set.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, Subset, random_split

from eeg_bci.data.adapters import BraindecodeLikeDataset, TensorDatasetFromBraindecode


class SplittableBraindecodeDataset(Protocol):
    """Protocol for Braindecode datasets that support metadata splitting."""

    def split(self, by: str | None = None, **kwargs: Any) -> dict[str, Any]: ...


class SizedDataset(Protocol):
    def __len__(self) -> int: ...


def split_train_test(
    dataset: BraindecodeLikeDataset,
    dataset_cfg: DictConfig,
    *,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into training and final test sets."""

    split_cfg = dataset_cfg.get("split", None)
    strategy = str(split_cfg.get("strategy", "random")) if split_cfg is not None else "random"

    if strategy == "description":
        return split_by_description(dataset, split_cfg)
    if strategy == "random":
        test_size = (
            float(split_cfg.get("test_size", dataset_cfg.get("test_size", 0.2)))
            if split_cfg is not None
            else float(dataset_cfg.get("test_size", 0.2))
        )
        return split_random_train_test(dataset, test_size=test_size, seed=seed)

    raise ValueError(f"Unsupported split strategy '{strategy}'.")


def split_train_eval(
    dataset: BraindecodeLikeDataset,
    dataset_cfg: DictConfig,
    *,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Backward-compatible alias for `split_train_test`."""

    return split_train_test(dataset, dataset_cfg, seed=seed)


def split_random_train_test(
    dataset: BraindecodeLikeDataset,
    *,
    test_size: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Create a deterministic random train/test split for smoke-test datasets."""

    tensor_dataset = TensorDatasetFromBraindecode(dataset)
    train_len, test_len = _split_lengths(len(tensor_dataset), holdout_size=test_size)
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(
        tensor_dataset,
        [train_len, test_len],
        generator=generator,
    )
    return train_set, test_set


def split_train_valid(
    train_set: Dataset,
    *,
    valid_size: float,
    seed: int,
    shuffle: bool,
) -> tuple[Dataset, Dataset]:
    """Split a training set into inner-training and validation subsets.

    For EEG/time-series experiments, prefer `shuffle=False` so nearby correlated
    windows are not randomly mixed across training and validation.
    """

    sized_train_set = cast(SizedDataset, train_set)
    train_len, valid_len = _split_lengths(len(sized_train_set), holdout_size=valid_size)

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        inner_train_set, valid_set = random_split(
            train_set,
            [train_len, valid_len],
            generator=generator,
        )
        return inner_train_set, valid_set

    indices = list(range(len(sized_train_set)))
    inner_train_indices = indices[:train_len]
    valid_indices = indices[train_len:]
    return Subset(train_set, inner_train_indices), Subset(train_set, valid_indices)


def split_by_description(
    dataset: BraindecodeLikeDataset, split_cfg: DictConfig
) -> tuple[Dataset, Dataset]:
    """Split using Braindecode dataset description metadata.

    Braindecode datasets carry a `description` table and expose `split()`,
    allowing protocol-aware splits by fields such as `subject`, `session`, or
    `run`. For BCI IV 2a, the default config uses `session` with `0train` and
    `1test`.
    """

    column = str(split_cfg.column)
    train_key = str(split_cfg.train_key)
    test_key = str(split_cfg.eval_key)

    if not hasattr(dataset, "split"):
        raise TypeError("Description-based splitting requires a Braindecode dataset with split().")

    splittable = cast(SplittableBraindecodeDataset, dataset)
    splits = splittable.split(column)
    missing_keys = [key for key in (train_key, test_key) if key not in splits]
    if missing_keys:
        available = ", ".join(sorted(splits))
        missing = ", ".join(missing_keys)
        raise KeyError(
            f"Split key(s) not found for column '{column}': {missing}. "
            f"Available keys: {available}."
        )

    return (
        TensorDatasetFromBraindecode(splits[train_key]),
        TensorDatasetFromBraindecode(splits[test_key]),
    )


def _split_lengths(total_len: int, *, holdout_size: float) -> tuple[int, int]:
    if total_len < 2:
        raise ValueError("At least two samples are required to create a split.")
    if not 0.0 < holdout_size < 1.0:
        raise ValueError("Holdout size must be between 0 and 1.")

    train_len = int(round(total_len * (1.0 - holdout_size)))
    train_len = min(max(train_len, 1), total_len - 1)
    holdout_len = total_len - train_len
    return train_len, holdout_len
