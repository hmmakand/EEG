"""Leakage-safe LOSO PyG DataLoaders for the CSD alpha-band wPLI combination.

Node features come from the ``csd`` variant (8 selected columns/electrode, computed
on the CSD-transformed signal); edges come from the ``wpli_csd`` variant,
restricted to the alpha band only. This is the CSD counterpart of
:mod:`without_csd_alpha_wpli` -- same connectivity method and band, different
signal source for both nodes and edges. Graphs are returned as sparse
``torch_geometric.data.Data`` objects, matching the without-CSD combination.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from src.datautils.graphdataversiontwo.saved_dataset import SavedDataset
from src.datautils.graphdataversiontwo.saved_dataset import load_dataset as _load_saved_dataset

from .loso_split import (
    FeatureNormalization,
    GraphDataLoaderConfig,
    LosoDataLoaderBundle,
    _graph_indices_for_subjects,
    create_loso_splits,
)
from .validation import (
    SOURCE_NODE_FEATURE_NAMES, select_node_features, selected_node_indices,
    DATASET_DIR,
    EXPECTED_EDGES,
    EXPECTED_GRAPHS,
    EXPECTED_NODES,
    EXPECTED_NODE_FEATURES,
    EXPECTED_SUBJECT_IDS,
    BAND_NAMES,
    NODE_FEATURE_NAMES,
    alpha_band_index,
)

NODE_VARIANT: Final[str] = "csd"
EDGE_VARIANT: Final[str] = "wpli_csd"
BAND_NAME: Final[str] = "alpha"


def _subject_ids(dataset: SavedDataset) -> np.ndarray:
    return np.asarray([sample["subject"] for sample in dataset.samples], dtype=np.int64)


def validate_dataset(dataset: SavedDataset) -> None:
    """Validate the fixed schema this combination assumes.

    Mirrors :func:`without_csd_alpha_wpli.validate_dataset` but pinned to the
    ``csd`` node variant and ``wpli_csd`` edge variant.
    """

    metadata = dataset.metadata
    if metadata.get("dataset") != "Liu2024":
        raise ValueError("Unexpected dataset identity in metadata")
    if metadata.get("n_samples") != EXPECTED_GRAPHS:
        raise ValueError(
            f"Expected {EXPECTED_GRAPHS} samples; metadata reports {metadata.get('n_samples')!r}"
        )
    if len(dataset) != EXPECTED_GRAPHS:
        raise ValueError(f"Loaded dataset has {len(dataset)} samples; expected {EXPECTED_GRAPHS}")
    selected_node_indices(dataset)
    if tuple(metadata.get("node_feature_names", ())) != SOURCE_NODE_FEATURE_NAMES:
        raise ValueError("Unexpected node-feature names or order")
    if tuple(metadata.get("band_names", ())) != BAND_NAMES:
        raise ValueError("Unexpected band names or order")
    if metadata.get("class_mapping") != {"left_hand": 0, "right_hand": 1}:
        raise ValueError("metadata class_mapping must map left_hand to 0 and right_hand to 1")

    if NODE_VARIANT not in dataset.nodes:
        raise ValueError(f"Dataset is missing node variant {NODE_VARIANT!r}")
    if EDGE_VARIANT not in dataset.edges:
        raise ValueError(f"Dataset is missing edge variant {EDGE_VARIANT!r}")

    node_features = dataset.nodes[NODE_VARIANT]
    expected_node_shape = (EXPECTED_GRAPHS, EXPECTED_NODES, len(SOURCE_NODE_FEATURE_NAMES))
    if node_features.shape != expected_node_shape:
        raise ValueError(
            f"{NODE_VARIANT} node features have shape {node_features.shape}; "
            f"expected {expected_node_shape}"
        )
    if not np.isfinite(node_features).all():
        raise ValueError(f"{NODE_VARIANT} node features contain non-finite values")

    edge_features = dataset.edges[EDGE_VARIANT]
    expected_edge_shape = (EXPECTED_GRAPHS, EXPECTED_EDGES, len(BAND_NAMES))
    if edge_features.shape != expected_edge_shape:
        raise ValueError(
            f"{EDGE_VARIANT} edge features have shape {edge_features.shape}; "
            f"expected {expected_edge_shape}"
        )
    if not np.isfinite(edge_features).all():
        raise ValueError(f"{EDGE_VARIANT} edge features contain non-finite values")

    if dataset.edge_index.shape != (2, EXPECTED_EDGES):
        raise ValueError(f"edge_index has shape {dataset.edge_index.shape}; expected (2, {EXPECTED_EDGES})")

    from src.datautils.graphdataversiontwo.edge_features import build_edge_index
    if not np.array_equal(dataset.edge_index, build_edge_index()):
        raise ValueError("Unexpected edge ordering")
    if np.any(edge_features < 0) or np.any(edge_features > 1):
        raise ValueError("wPLI values must lie in [0, 1]")
    if not np.allclose(edge_features[:, :406], edge_features[:, 406:], rtol=0, atol=1e-7):
        raise ValueError("Reverse edge weights differ")
    labels = np.asarray(dataset.labels)
    if labels.shape != (EXPECTED_GRAPHS,):
        raise ValueError(f"labels has shape {labels.shape}; expected ({EXPECTED_GRAPHS},)")
    if set(map(int, np.unique(labels))) != {0, 1}:
        raise ValueError("Dataset labels must contain exactly the classes 0 and 1")

    subject_ids = _subject_ids(dataset)
    subjects, subject_counts = np.unique(subject_ids, return_counts=True)
    if tuple(map(int, subjects)) != EXPECTED_SUBJECT_IDS:
        raise ValueError("Dataset must contain subjects 1 through 50 exactly")
    if not np.array_equal(subject_counts, np.full(50, 40, dtype=np.int64)):
        raise ValueError("Each subject must contribute exactly 40 trials")


def load_dataset(path: str | Path = DATASET_DIR) -> SavedDataset:
    """Load and fully validate this combination's canonical dataset."""

    dataset = _load_saved_dataset(path)
    validate_dataset(dataset)
    return dataset


class _CsdAlphaWpliGraphSubset(Dataset):
    """Build sparse PyG graphs: CSD nodes, alpha-band CSD wPLI edges.

    Reads ``nodes["csd"]``/``edges["wpli_csd"]``/``labels`` directly from the
    saved dataset's underlying memory maps and slices the alpha-band column
    out of the 5-band edge array.
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
            select_node_features(self.dataset, NODE_VARIANT, graph_index), dtype=np.float32
        )
        if node_features.shape != (EXPECTED_NODES, EXPECTED_NODE_FEATURES):
            raise ValueError(
                "Unexpected node_features shape for the CSD alpha-wPLI "
                f"combination: {node_features.shape}"
            )

        # torch.tensor copies the memory-mapped read-only row so the result is
        # safe to move, batch, or modify.
        x = torch.tensor(node_features, dtype=torch.float32)
        if self.normalization.mode == "zscore":
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
    dataset: SavedDataset, *, epsilon: float = 1e-8,
    graph_indices: np.ndarray | None = None, mode: str = "zscore",
) -> FeatureNormalization:
    """Fit shared eight-column statistics from explicit training indices only.

    Disabled mode returns an explicit empty state without reading feature values.
    Both signal variants use this definition independently.
    """
    if mode not in ("none", "zscore"):
        raise ValueError("node_normalization must be none or zscore")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if mode == "none":
        return FeatureNormalization({}, {}, mode="none", epsilon=epsilon)
    if graph_indices is None:
        raise ValueError("zscore requires explicit training graph_indices")
    indices = np.asarray(graph_indices)
    if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(dataset))
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("Training graph_indices must be unique valid integer indices")
    values = np.asarray(select_node_features(dataset, NODE_VARIANT, indices), dtype=np.float64)
    if values.shape[1:] != (EXPECTED_NODES, EXPECTED_NODE_FEATURES) or not np.isfinite(values).all():
        raise ValueError("Training node features must be finite with shape (N, 29, 8)")
    mean = torch.tensor(values.mean(axis=(0, 1)), dtype=torch.float32)
    std = torch.tensor(np.maximum(values.std(axis=(0, 1)), epsilon), dtype=torch.float32)
    subjects = {int(sample["subject"]) for sample in dataset.samples}
    return FeatureNormalization({s: mean for s in subjects}, {s: std for s in subjects},
                                mode="zscore", fit_graph_indices=tuple(map(int, indices)), epsilon=epsilon)


def _make_loader(
    dataset: SavedDataset,
    graph_indices: np.ndarray,
    normalization: FeatureNormalization,
    config: GraphDataLoaderConfig,
    *,
    band_index: int,
    shuffle: bool,
) -> DataLoader:
    subset = _CsdAlphaWpliGraphSubset(
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
    subject_ids = _subject_ids(resolved_dataset)
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
        graph_indices=split.train_graph_indices, mode=resolved_config.node_normalization,
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


def _create_indexed_dataloaders(
    dataset: SavedDataset,
    train_graph_indices: np.ndarray,
    evaluation_graph_indices: np.ndarray,
    config: GraphDataLoaderConfig,
    *,
    dataset_validated: bool = False,
    normalization: FeatureNormalization | None = None,
) -> tuple[DataLoader, DataLoader, FeatureNormalization]:
    """Build a loader pair with optional normalization fitted on training indices.

    A supplied state must match both the selected mode and training scope.
    """

    if not dataset_validated:
        validate_dataset(dataset)
    band_index = alpha_band_index(dataset)
    if normalization is None:
        normalization = fit_feature_normalization(
            dataset, epsilon=config.normalization_epsilon,
            graph_indices=train_graph_indices, mode=config.node_normalization
        )
    if normalization.mode != config.node_normalization:
        raise ValueError("Normalization mode does not match loader configuration")
    if normalization.mode == "zscore" and set(normalization.fit_graph_indices) != set(map(int, train_graph_indices)):
        raise ValueError("Normalization must be fitted on this fold's training indices")
    train_loader = _make_loader(
        dataset,
        train_graph_indices,
        normalization,
        config,
        band_index=band_index,
        shuffle=True,
    )
    evaluation_loader = _make_loader(
        dataset,
        evaluation_graph_indices,
        normalization,
        config,
        band_index=band_index,
        shuffle=False,
    )
    return train_loader, evaluation_loader, normalization


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

    subject_ids = _subject_ids(dataset)
    train_graph_indices = _graph_indices_for_subjects(subject_ids, train_subject_ids)
    validation_graph_indices = _graph_indices_for_subjects(
        subject_ids, validation_subject_ids
    )
    return _create_indexed_dataloaders(
        dataset,
        train_graph_indices,
        validation_graph_indices,
        config,
        dataset_validated=dataset_validated,
    )


def create_within_subject_dataloaders(
    *,
    dataset: SavedDataset,
    train_graph_indices: np.ndarray,
    evaluation_graph_indices: np.ndarray,
    config: GraphDataLoaderConfig,
    dataset_validated: bool = False,
    normalization: FeatureNormalization | None = None,
) -> tuple[DataLoader, DataLoader, FeatureNormalization]:
    """Build train/evaluation loaders for one within-subject diagnostic fold.

    Takes raw graph indices directly -- a within-subject fold has no
    subject-ID tuple to resolve, since its indices are already one
    subject's own trials (see
    ``src.ourexperimentversionfive.data.within_subject_split``). Pass a
    pre-fit ``normalization`` (see :func:`_create_indexed_dataloaders`) to
    avoid refitting it on every one of a run's many folds.
    """

    return _create_indexed_dataloaders(
        dataset,
        train_graph_indices,
        evaluation_graph_indices,
        config,
        dataset_validated=dataset_validated,
        normalization=normalization,
    )
