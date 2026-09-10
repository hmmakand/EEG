"""Strict validation for experiment three's fixed broadcast-11 dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from src.datautils.PlvLiu2024Broadcast11 import (
    DEFAULT_OUTPUT_DIR,
    PlvLiu2024GraphDataset,
)


DATASET_DIR: Final[Path] = DEFAULT_OUTPUT_DIR
DATASET_VARIANT: Final[str] = "manuscript_broadcast_11_v3"
GRAPH_FORMAT_VERSION: Final[int] = 3
EXPECTED_GRAPHS: Final[int] = 2_000
EXPECTED_NODES: Final[int] = 29
EXPECTED_NODE_FEATURES: Final[int] = 11
EXPECTED_GRAPH_FEATURES: Final[int] = 0
EXPECTED_SUBJECT_IDS: Final[tuple[int, ...]] = tuple(range(1, 51))

# These literals intentionally live in the experiment package.  Validation must
# fail if a future generator silently changes the model-input contract.
LOCAL_NODE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "MAV",
    "STD",
    "PSD",
    "betweenness",
)
BROADCAST_NODE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "PersistenceEntropy_0",
    "landscape1_0",
    "landscape1_1",
    "landscape2_0",
    "landscape2_1",
    "betti_0",
    "betti_1",
)
NODE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    *LOCAL_NODE_FEATURE_NAMES,
    *BROADCAST_NODE_FEATURE_NAMES,
)
BROADCAST_SOURCE_FILENAME: Final[str] = "broadcast_source_features.npy"


def _validate_metadata(dataset: PlvLiu2024GraphDataset) -> None:
    metadata = dataset.metadata
    expected_scalars: dict[str, object] = {
        "graph_format_version": GRAPH_FORMAT_VERSION,
        "dataset_variant": DATASET_VARIANT,
        "n_graphs": EXPECTED_GRAPHS,
        "n_nodes": EXPECTED_NODES,
        "n_node_features": EXPECTED_NODE_FEATURES,
        "n_graph_features": EXPECTED_GRAPH_FEATURES,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_scalars.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Unexpected experiment-three dataset schema: {details}")

    if tuple(metadata.get("node_feature_names", ())) != NODE_FEATURE_NAMES:
        raise ValueError(
            "Unexpected node-feature names or order; expected "
            f"{NODE_FEATURE_NAMES!r}"
        )
    if tuple(metadata.get("graph_feature_names", ())) != ():
        raise ValueError("Experiment three does not accept separate graph features")
    if (
        tuple(metadata.get("broadcast_source_feature_names", ()))
        != BROADCAST_NODE_FEATURE_NAMES
    ):
        raise ValueError("Unexpected broadcast-source feature names or order")

    feature_scopes = metadata.get("feature_scopes")
    if not isinstance(feature_scopes, dict):
        raise ValueError("Dataset metadata is missing feature_scopes")
    if feature_scopes.get("global_features_broadcast_to_nodes") is not True:
        raise ValueError("Dataset does not declare global features broadcast to nodes")
    if tuple(feature_scopes.get("broadcast_columns", ())) != tuple(range(4, 11)):
        raise ValueError("Expected broadcast columns 4 through 10")

    metadata_subjects = tuple(sorted(map(int, metadata.get("subjects", ()))))
    if metadata_subjects != EXPECTED_SUBJECT_IDS:
        raise ValueError("Dataset metadata must list subjects 1 through 50 exactly")


def _validate_array(
    name: str,
    array: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
) -> None:
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if array.dtype != dtype:
        raise ValueError(f"{name} has dtype {array.dtype}; expected {dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")


def load_broadcast_source_features(dataset: PlvLiu2024GraphDataset) -> np.ndarray:
    """Memory-map the seven graph scalars retained for broadcast auditing."""

    source_path = dataset.data_dir / BROADCAST_SOURCE_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Experiment-three dataset is missing {BROADCAST_SOURCE_FILENAME}: "
            f"{source_path}"
        )
    return np.load(source_path, mmap_mode="r")


def validate_dataset(dataset: PlvLiu2024GraphDataset) -> None:
    """Validate the complete fixed schema, values, and exact broadcast rule.

    This deliberately checks the full arrays, rather than a sample, before a
    training run can consume them.
    """

    _validate_metadata(dataset)
    arrays = dataset.arrays
    _validate_array(
        "node_features",
        arrays.node_features,
        shape=(EXPECTED_GRAPHS, EXPECTED_NODES, EXPECTED_NODE_FEATURES),
        dtype=np.dtype(np.float32),
    )
    _validate_array(
        "plv_matrices",
        arrays.plv_matrices,
        shape=(EXPECTED_GRAPHS, EXPECTED_NODES, EXPECTED_NODES),
        dtype=np.dtype(np.float32),
    )
    _validate_array(
        "adjacency_matrices",
        arrays.adjacency_matrices,
        shape=(EXPECTED_GRAPHS, EXPECTED_NODES, EXPECTED_NODES),
        dtype=np.dtype(np.float32),
    )
    for name in ("labels", "subject_ids", "trial_indices", "window_indices"):
        _validate_array(
            name,
            getattr(arrays, name),
            shape=(EXPECTED_GRAPHS,),
            dtype=np.dtype(np.int64),
        )
    if arrays.graph_features is not None:
        raise ValueError("Experiment-three dataset unexpectedly has graph_features")

    if set(map(int, np.unique(arrays.labels))) != {0, 1}:
        raise ValueError("Dataset labels must contain exactly the classes 0 and 1")
    subjects, subject_counts = np.unique(arrays.subject_ids, return_counts=True)
    if tuple(map(int, subjects)) != EXPECTED_SUBJECT_IDS:
        raise ValueError("Dataset arrays must contain subjects 1 through 50 exactly")
    if not np.array_equal(subject_counts, np.full(50, 40, dtype=np.int64)):
        raise ValueError("Each subject must contribute exactly 40 graphs")

    source_features = load_broadcast_source_features(dataset)
    _validate_array(
        BROADCAST_SOURCE_FILENAME,
        source_features,
        shape=(EXPECTED_GRAPHS, len(BROADCAST_NODE_FEATURE_NAMES)),
        dtype=np.dtype(np.float32),
    )
    broadcast_columns = arrays.node_features[:, :, len(LOCAL_NODE_FEATURE_NAMES) :]
    expected_broadcast = np.broadcast_to(
        source_features[:, None, :], broadcast_columns.shape
    )
    if not np.array_equal(broadcast_columns, expected_broadcast):
        raise ValueError(
            "Node-feature columns 4 through 10 are not exact per-graph broadcasts"
        )


def load_dataset() -> PlvLiu2024GraphDataset:
    """Load and fully validate experiment three's one canonical dataset."""

    dataset = PlvLiu2024GraphDataset(DATASET_DIR)
    validate_dataset(dataset)
    return dataset

