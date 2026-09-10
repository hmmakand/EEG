"""Leakage-safe LOSO PyG DataLoaders for one Graph-Liu2024-VersionOne combination.

Node features come from the ``without_csd`` variant (6 columns/electrode);
edges come from the ``wpli_without_csd`` variant, restricted to the alpha
band only (one scalar weight per directed electrode pair). Graphs are
returned as sparse ``torch_geometric.data.Data`` objects -- the saved
dataset carries no precomputed adjacency matrix, so there is nothing dense
to threshold or binarize.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from src.datautils.graphdataversionone.saved_dataset import SavedDataset

from .loso_split import (
    FeatureNormalization,
    GraphDataLoaderConfig,
    LosoDataLoaderBundle,
    _graph_indices_for_subjects,
    create_loso_splits,
)
from .validation import (
    BAND_NAME,
    EDGE_VARIANT,
    EXPECTED_NODE_FEATURES,
    EXPECTED_NODES,
    NODE_VARIANT,
    alpha_band_index,
    load_dataset,
    validate_dataset,
)


class _AlphaWpliGraphSubset(Dataset):
    """Build sparse PyG graphs: without-CSD nodes, alpha-band wPLI edges.

    Reads ``nodes["without_csd"]``/``edges["wpli_without_csd"]``/``labels``
    directly from the saved dataset's underlying memory maps and slices the
    alpha-band column out of the 5-band edge array.
    """

    def __init__(
        self,
        dataset: SavedDataset,
        graph_indices: np.ndarray,
        normalization: FeatureNormalization,
        *,
        band_index: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.graph_indices = np.asarray(graph_indices, dtype=np.int64)
        self.normalization = normalization
        self.band_index = band_index
        # Shared across every graph: same 29 electrodes, same directed pairs.
        self.edge_index = torch.from_numpy(
            np.array(dataset.edge_index, dtype=np.int64, copy=True)
        )

    def len(self) -> int:
        return len(self.graph_indices)

    def get(self, idx: int) -> Data:
        graph_index = int(self.graph_indices[idx])
        node_features = np.asarray(
            self.dataset.nodes[NODE_VARIANT][graph_index], dtype=np.float32
        )
        if node_features.shape != (EXPECTED_NODES, EXPECTED_NODE_FEATURES):
            raise ValueError(
                "Unexpected node_features shape for the without-CSD alpha-wPLI "
                f"combination: {node_features.shape}"
            )

        # torch.tensor copies the memory-mapped read-only row so the result is
        # safe to move, batch, or modify.
        x = torch.tensor(node_features, dtype=torch.float32)
        subject = int(self.dataset.samples[graph_index]["subject"])
        mean = self.normalization.mean_by_subject[subject]
        standard_deviation = self.normalization.standard_deviation_by_subject[subject]
        x = (x - mean) / standard_deviation
        if not torch.isfinite(x).all():
            raise ValueError("Normalization produced non-finite node features")

        edge_weights = np.asarray(
            self.dataset.edges[EDGE_VARIANT][graph_index, :, self.band_index],
            dtype=np.float32,
        )
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)
        if not torch.isfinite(edge_attr).all():
            raise ValueError("Alpha-band edge weights contain non-finite values")

        label = int(self.dataset.labels[graph_index])
        return Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.long),
        )


def fit_feature_normalization(
    dataset: SavedDataset,
    *,
    epsilon: float = 1e-8,
) -> FeatureNormalization:
    """Fit one z-scoring statistic per subject from that subject's own trials.

    Every subject present in ``dataset`` gets its own six-column mean/std,
    computed only from its own ``without_csd`` node features -- independent
    of which LOSO fold or split a subject ends up in, so this can be (and
    is) fit once per dataset rather than once per fold.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    subject_ids = np.asarray(
        [sample["subject"] for sample in dataset.samples], dtype=np.int64
    )
    node_features = dataset.nodes[NODE_VARIANT]
    mean_by_subject: dict[int, torch.Tensor] = {}
    standard_deviation_by_subject: dict[int, torch.Tensor] = {}
    for subject in np.unique(subject_ids):
        indices = np.flatnonzero(subject_ids == subject)
        subject_features = np.asarray(node_features[indices], dtype=np.float64)
        if subject_features.shape[1:] != (EXPECTED_NODES, EXPECTED_NODE_FEATURES):
            raise ValueError(
                "Subject node features do not satisfy the fixed (29, 6) contract"
            )
        if not np.isfinite(subject_features).all():
            raise ValueError(f"Subject {int(subject)} node features contain non-finite values")
        mean = subject_features.mean(axis=(0, 1))
        standard_deviation = np.maximum(subject_features.std(axis=(0, 1)), epsilon)
        mean_by_subject[int(subject)] = torch.tensor(mean, dtype=torch.float32)
        standard_deviation_by_subject[int(subject)] = torch.tensor(
            standard_deviation, dtype=torch.float32
        )
    return FeatureNormalization(
        mean_by_subject=mean_by_subject,
        standard_deviation_by_subject=standard_deviation_by_subject,
    )


def _make_loader(
    dataset: SavedDataset,
    graph_indices: np.ndarray,
    normalization: FeatureNormalization,
    config: GraphDataLoaderConfig,
    *,
    band_index: int,
    shuffle: bool,
) -> DataLoader:
    subset = _AlphaWpliGraphSubset(
        dataset,
        graph_indices,
        normalization,
        band_index=band_index,
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
    dataset: SavedDataset | None = None,
    dataset_validated: bool = False,
) -> LosoDataLoaderBundle:
    """Create normalized loaders for one held-out subject.

    Validation is safe by default. An all-fold orchestrator may call
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

    band_index = alpha_band_index(resolved_dataset)
    subject_ids = np.asarray(
        [sample["subject"] for sample in resolved_dataset.samples], dtype=np.int64
    )
    splits = create_loso_splits(
        subject_ids,
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
        epsilon=resolved_config.normalization_epsilon,
    )
    return LosoDataLoaderBundle(
        train=_make_loader(
            resolved_dataset,
            split.train_graph_indices,
            normalization,
            resolved_config,
            band_index=band_index,
            shuffle=True,
        ),
        validation=_make_loader(
            resolved_dataset,
            split.validation_graph_indices,
            normalization,
            resolved_config,
            band_index=band_index,
            shuffle=False,
        ),
        test=_make_loader(
            resolved_dataset,
            split.test_graph_indices,
            normalization,
            resolved_config,
            band_index=band_index,
            shuffle=False,
        ),
        split=split,
        normalization=normalization,
    )


def create_group_dataloaders(
    *,
    dataset: SavedDataset,
    train_subject_ids: tuple[int, ...],
    validation_subject_ids: tuple[int, ...],
    config: GraphDataLoaderConfig,
    dataset_validated: bool = False,
) -> tuple[DataLoader, DataLoader, FeatureNormalization]:
    """Build train/validation loaders for an arbitrary pair of subject groups.

    Unlike :func:`create_loso_dataloaders`, there is no held-out test subject
    concept here -- this is what powers the inner-CV hyperparameter search,
    which only ever needs a train/validation pair drawn from one outer
    fold's development subjects.
    """

    if not dataset_validated:
        validate_dataset(dataset)

    band_index = alpha_band_index(dataset)
    subject_ids = np.asarray(
        [sample["subject"] for sample in dataset.samples], dtype=np.int64
    )
    train_graph_indices = _graph_indices_for_subjects(subject_ids, train_subject_ids)
    validation_graph_indices = _graph_indices_for_subjects(
        subject_ids, validation_subject_ids
    )
    normalization = fit_feature_normalization(
        dataset, epsilon=config.normalization_epsilon
    )
    train_loader = _make_loader(
        dataset,
        train_graph_indices,
        normalization,
        config,
        band_index=band_index,
        shuffle=True,
    )
    validation_loader = _make_loader(
        dataset,
        validation_graph_indices,
        normalization,
        config,
        band_index=band_index,
        shuffle=False,
    )
    return train_loader, validation_loader, normalization
