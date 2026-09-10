"""Leakage-safe LOSO DataLoaders for experiment three's fixed dataset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.datautils.PlvLiu2024Broadcast11 import PlvLiu2024GraphDataset

from .validation import EXPECTED_NODE_FEATURES, EXPECTED_NODES, load_dataset, validate_dataset


@dataclass(frozen=True)
class GraphDataLoaderConfig:
    """Runtime and validation-split options for one LOSO fold."""

    batch_size: int = 32
    validation_subjects: int = 5
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    seed: int = 42
    normalization_epsilon: float = 1e-8
    plv_threshold: float = 0.3
    """Manuscript's ablation-selected graph rule: retain and binarize PLV
    edges >= this threshold. See EEGGCN1Config's docstring for context."""

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.validation_subjects <= 0:
            raise ValueError("validation_subjects must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        if not 0 <= self.plv_threshold <= 1:
            raise ValueError("plv_threshold must be between zero and one")


@dataclass(frozen=True)
class LosoGraphSplit:
    """Graph indices and subject IDs assigned to one LOSO fold."""

    train_graph_indices: np.ndarray
    validation_graph_indices: np.ndarray
    test_graph_indices: np.ndarray
    train_subject_ids: tuple[int, ...]
    validation_subject_ids: tuple[int, ...]
    test_subject_id: int
    fold: int


@dataclass(frozen=True)
class FeatureNormalization:
    """Per-column statistics fitted exclusively on training graph nodes."""

    mean: torch.Tensor
    standard_deviation: torch.Tensor


@dataclass(frozen=True)
class LosoDataLoaderBundle:
    """Loaders plus the split and preprocessing provenance for one fold."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    split: LosoGraphSplit
    normalization: FeatureNormalization


class _DenseGraphSubset(Dataset):
    """Select saved graphs as dense (x, adj, y) tensors for EEGGCN1.

    Reads ``node_features``/``plv_matrices``/``labels`` directly from the
    dataset's underlying arrays rather than going through
    ``PlvLiu2024GraphDataset.__getitem__`` (which builds a sparse PyG
    ``Data`` object this dense model does not use). The adjacency is derived
    fresh from the full PLV matrix by thresholding and binarizing, matching
    the manuscript's ablation-selected graph construction -- not the top-k
    weighted adjacency saved as ``adjacency_matrices.npy``.
    """

    def __init__(
        self,
        dataset: PlvLiu2024GraphDataset,
        graph_indices: np.ndarray,
        normalization: FeatureNormalization,
        *,
        plv_threshold: float,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.graph_indices = np.asarray(graph_indices, dtype=np.int64)
        self.normalization = normalization
        self.plv_threshold = plv_threshold

    def __len__(self) -> int:
        return len(self.graph_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_index = int(self.graph_indices[idx])
        node_features = np.asarray(
            self.dataset.arrays.node_features[graph_index], dtype=np.float32
        )
        if node_features.shape != (EXPECTED_NODES, EXPECTED_NODE_FEATURES):
            raise ValueError(
                "Unexpected node_features shape while loading experiment-three "
                f"data: {node_features.shape}"
            )
        plv_matrix = np.asarray(
            self.dataset.arrays.plv_matrices[graph_index], dtype=np.float32
        )
        label = int(self.dataset.arrays.labels[graph_index])

        # torch.tensor copies the memory-mapped read-only row so the result is
        # safe to move, batch, or modify, matching PlvLiu2024Broadcast11's own
        # create_pyg_graph convention for the same underlying arrays.
        x = torch.tensor(node_features, dtype=torch.float32)
        x = (x - self.normalization.mean) / self.normalization.standard_deviation
        if not torch.isfinite(x).all():
            raise ValueError("Normalization produced non-finite node features")

        adjacency = torch.from_numpy(
            (plv_matrix >= self.plv_threshold).astype(np.float32)
        )
        return x, adjacency, torch.tensor(label, dtype=torch.long)


def _graph_indices_for_subjects(
    subject_ids: np.ndarray, selected_subjects: tuple[int, ...]
) -> np.ndarray:
    return np.flatnonzero(np.isin(subject_ids, selected_subjects)).astype(
        np.int64, copy=False
    )


def _validate_split(split: LosoGraphSplit, subject_ids: np.ndarray) -> None:
    graph_sets = (
        set(map(int, split.train_graph_indices)),
        set(map(int, split.validation_graph_indices)),
        set(map(int, split.test_graph_indices)),
    )
    if any(
        graph_sets[left] & graph_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError("LOSO partitions contain overlapping graph indices")
    if set.union(*graph_sets) != set(range(len(subject_ids))):
        raise RuntimeError("LOSO partitions do not cover every graph exactly once")

    subject_sets = (
        set(split.train_subject_ids),
        set(split.validation_subject_ids),
        {split.test_subject_id},
    )
    if any(
        subject_sets[left] & subject_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError("LOSO partitions leak subjects")

    observed_subject_sets = tuple(
        set(map(int, np.unique(subject_ids[indices])))
        for indices in (
            split.train_graph_indices,
            split.validation_graph_indices,
            split.test_graph_indices,
        )
    )
    if observed_subject_sets != subject_sets:
        raise RuntimeError("LOSO graph indices do not match their assigned subjects")


def create_loso_splits(
    subject_ids: np.ndarray,
    *,
    validation_subjects: int = 5,
    seed: int = 42,
) -> tuple[LosoGraphSplit, ...]:
    """Create one reproducible, subject-disjoint fold per observed subject."""

    raw_values = np.asarray(subject_ids)
    if raw_values.ndim != 1 or len(raw_values) == 0:
        raise ValueError("subject_ids must be a non-empty one-dimensional array")
    if not np.issubdtype(raw_values.dtype, np.integer):
        raise TypeError("subject_ids must have an integer dtype")
    values = raw_values.astype(np.int64, copy=False)
    subjects = np.unique(values)
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for train/validation/test")
    if not 1 <= validation_subjects <= len(subjects) - 2:
        raise ValueError(
            "validation_subjects must leave one test and at least one training subject"
        )

    splits: list[LosoGraphSplit] = []
    for fold, test_subject in enumerate(subjects):
        development_subjects = subjects[subjects != test_subject]
        # Including the held-out ID makes each validation draw deterministic but
        # independent across LOSO folds.
        shuffled = np.random.default_rng(seed + int(test_subject)).permutation(
            development_subjects
        )
        validation_ids = tuple(sorted(map(int, shuffled[:validation_subjects])))
        validation_set = set(validation_ids)
        train_ids = tuple(
            sorted(
                int(subject)
                for subject in development_subjects
                if int(subject) not in validation_set
            )
        )
        split = LosoGraphSplit(
            train_graph_indices=_graph_indices_for_subjects(values, train_ids),
            validation_graph_indices=_graph_indices_for_subjects(values, validation_ids),
            test_graph_indices=_graph_indices_for_subjects(
                values, (int(test_subject),)
            ),
            train_subject_ids=train_ids,
            validation_subject_ids=validation_ids,
            test_subject_id=int(test_subject),
            fold=fold,
        )
        _validate_split(split, values)
        splits.append(split)

    return tuple(splits)


def fit_feature_normalization(
    dataset: PlvLiu2024GraphDataset,
    train_graph_indices: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> FeatureNormalization:
    """Fit all 11 column statistics using training subjects only."""

    indices = np.asarray(train_graph_indices)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("Cannot fit normalization without training graph indices")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("train_graph_indices must have an integer dtype")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if np.any(indices < 0) or np.any(indices >= len(dataset)):
        raise IndexError("Training graph index is outside the dataset")

    training_features = np.asarray(
        dataset.arrays.node_features[indices.astype(np.int64, copy=False)],
        dtype=np.float64,
    )
    if training_features.shape[1:] != (EXPECTED_NODES, EXPECTED_NODE_FEATURES):
        raise ValueError(
            "Training node features do not satisfy the fixed (29, 11) contract"
        )
    if not np.isfinite(training_features).all():
        raise ValueError("Training node features contain non-finite values")
    mean = training_features.mean(axis=(0, 1))
    standard_deviation = training_features.std(axis=(0, 1))
    standard_deviation = np.maximum(standard_deviation, epsilon)
    return FeatureNormalization(
        mean=torch.tensor(mean, dtype=torch.float32),
        standard_deviation=torch.tensor(standard_deviation, dtype=torch.float32),
    )


def _make_loader(
    dataset: PlvLiu2024GraphDataset,
    graph_indices: np.ndarray,
    normalization: FeatureNormalization,
    config: GraphDataLoaderConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    subset = _DenseGraphSubset(
        dataset,
        graph_indices,
        normalization,
        plv_threshold=config.plv_threshold,
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        generator=generator if shuffle else None,
    )


def create_loso_dataloaders(
    *,
    test_subject_id: int,
    config: GraphDataLoaderConfig | None = None,
    dataset: PlvLiu2024GraphDataset | None = None,
    dataset_validated: bool = False,
) -> LosoDataLoaderBundle:
    """Create normalized loaders for one held-out subject.

    Validation is safe by default.  An all-fold orchestrator may call
    :func:`validate_dataset` once and then pass ``dataset_validated=True`` for
    each fold to avoid scanning the same immutable memory maps 50 times.
    """

    resolved_config = config or GraphDataLoaderConfig()
    if dataset is None:
        resolved_dataset = load_dataset()
    else:
        resolved_dataset = dataset
        if not dataset_validated:
            validate_dataset(resolved_dataset)

    splits = create_loso_splits(
        np.asarray(resolved_dataset.arrays.subject_ids),
        validation_subjects=resolved_config.validation_subjects,
        seed=resolved_config.seed,
    )
    split_by_subject = {split.test_subject_id: split for split in splits}
    if test_subject_id not in split_by_subject:
        available = tuple(split_by_subject)
        raise ValueError(
            f"Unknown test subject {test_subject_id}; available subjects: {available}"
        )
    split = split_by_subject[test_subject_id]
    normalization = fit_feature_normalization(
        resolved_dataset,
        split.train_graph_indices,
        epsilon=resolved_config.normalization_epsilon,
    )
    return LosoDataLoaderBundle(
        train=_make_loader(
            resolved_dataset,
            split.train_graph_indices,
            normalization,
            resolved_config,
            shuffle=True,
        ),
        validation=_make_loader(
            resolved_dataset,
            split.validation_graph_indices,
            normalization,
            resolved_config,
            shuffle=False,
        ),
        test=_make_loader(
            resolved_dataset,
            split.test_graph_indices,
            normalization,
            resolved_config,
            shuffle=False,
        ),
        split=split,
        normalization=normalization,
    )
