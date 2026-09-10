"""Fixed contract and loading for the without-CSD alpha-band wPLI combination.

Reads the shared ``Graph-Liu2024-VersionOne`` dataset built by
``src.datautils.graphdataversionone`` and pins this experiment to one node
variant / edge variant / band combination: ``without_csd`` node features and
``wpli_without_csd`` edges restricted to the alpha band. Other combinations
(other bands, methods, or the CSD variants) belong in sibling modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np

from src.datautils.graphdataversionone.config import OUTPUT_ROOT
from src.datautils.graphdataversionone.saved_dataset import SavedDataset
from src.datautils.graphdataversionone.saved_dataset import load_dataset as _load_saved_dataset


DATASET_DIR: Final[Path] = OUTPUT_ROOT
NODE_VARIANT: Final[str] = "without_csd"
EDGE_VARIANT: Final[str] = "wpli_without_csd"
BAND_NAME: Final[str] = "alpha"
BAND_NAMES: Final[tuple[str, ...]] = ("delta", "theta", "alpha", "beta", "gamma")
EXPECTED_GRAPHS: Final[int] = 2_000
EXPECTED_NODES: Final[int] = 29
EXPECTED_NODE_FEATURES: Final[int] = 6
EXPECTED_EDGES: Final[int] = 812
EXPECTED_SUBJECT_IDS: Final[tuple[int, ...]] = tuple(range(1, 51))
NODE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "delta_power_db",
    "theta_power_db",
    "alpha_power_db",
    "beta_power_db",
    "gamma_power_db",
    "spectral_entropy",
)


def alpha_band_index(dataset: SavedDataset) -> int:
    """Return the column selecting the alpha band within one edge variant."""

    band_names = list(dataset.metadata.get("band_names", ()))
    if band_names != list(BAND_NAMES):
        raise ValueError(f"Unexpected band_names order/content: {band_names!r}")
    return band_names.index(BAND_NAME)


def _subject_ids(dataset: SavedDataset) -> np.ndarray:
    return np.asarray([sample["subject"] for sample in dataset.samples], dtype=np.int64)


def validate_dataset(dataset: SavedDataset) -> None:
    """Validate the fixed schema this experiment's combination assumes.

    This checks only the contract this module relies on (dataset identity,
    the two selected variants, shapes, finiteness, and subject/class
    balance). Full cross-file consistency of the saved dataset is checked
    once by ``src.datautils.graphdataversionone.validation.validate_saved_dataset``.
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
    if tuple(metadata.get("node_feature_names", ())) != NODE_FEATURE_NAMES:
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
    expected_node_shape = (EXPECTED_GRAPHS, EXPECTED_NODES, EXPECTED_NODE_FEATURES)
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
