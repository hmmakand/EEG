"""Load and expose the versioned Liu2024 PLV graph arrays.

The canonical on-disk representation uses memory-mappable NumPy arrays for
node features, dense PLV, weighted adjacency, labels, and provenance. This
module loads those arrays and can materialize any row as a PyTorch Geometric
``Data`` object. It does not generate or overwrite graph datasets.

This is a self-contained copy of ``src.datautils.PlvLiu2024.saved_dataset``
kept so that ``PlvLiu2024Broadcast11`` has no import dependency on
``PlvLiu2024``. Keep in sync by hand if the original is ever fixed. Unlike the
original, this copy does not keep a ``PlvGraphConfig``-style attribute on the
dataset object -- nothing in this package or in ``ourexperimentversionthree``
reads it, and ``validate_pyg_graph`` here only ever needed the node count,
which is read straight from metadata instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .config import DEFAULT_OUTPUT_DIR
from .graph import GraphComponents, create_pyg_graph, validate_pyg_graph


BASE_REQUIRED_GRAPH_FILES = (
    "node_features.npy",
    "plv_matrices.npy",
    "adjacency_matrices.npy",
    "labels.npy",
    "subject_ids.npy",
    "trial_indices.npy",
    "window_indices.npy",
    "metadata.json",
)


@dataclass(frozen=True)
class GraphDatasetArrays:
    """Dense graph arrays and provenance stored in the saved dataset."""

    node_features: np.ndarray
    plv_matrices: np.ndarray
    adjacency_matrices: np.ndarray
    labels: np.ndarray
    subject_ids: np.ndarray
    trial_indices: np.ndarray
    window_indices: np.ndarray
    graph_features: np.ndarray | None = None


def required_graph_files(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Return required filenames for the recorded graph format."""

    version = int(metadata.get("graph_format_version", 0))
    if version < 1:
        raise ValueError(f"Unsupported graph format version: {version}")
    return BASE_REQUIRED_GRAPH_FILES


def validate_required_graph_files(
    data_dir: Path, metadata: dict[str, Any] | None = None
) -> None:
    """Raise when a required saved graph file is missing."""

    resolved_metadata = metadata or load_graph_metadata(data_dir)
    missing = [
        name for name in required_graph_files(resolved_metadata)
        if not (data_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete graph dataset at {data_dir}; missing: {missing}")


def load_graph_metadata(data_dir: str | Path) -> dict[str, Any]:
    """Load graph-generation metadata from JSON."""

    resolved = Path(data_dir).expanduser().resolve()
    with (resolved / "metadata.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def load_graph_arrays(
    data_dir: str | Path,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
) -> GraphDatasetArrays:
    """Memory-map all canonical graph arrays from a saved dataset directory."""

    resolved = Path(data_dir).expanduser().resolve()
    metadata = load_graph_metadata(resolved)
    validate_required_graph_files(resolved, metadata)
    graph_features_path = resolved / "graph_features.npy"
    return GraphDatasetArrays(
        node_features=np.load(resolved / "node_features.npy", mmap_mode=mmap_mode),
        plv_matrices=np.load(resolved / "plv_matrices.npy", mmap_mode=mmap_mode),
        adjacency_matrices=np.load(
            resolved / "adjacency_matrices.npy", mmap_mode=mmap_mode
        ),
        labels=np.load(resolved / "labels.npy", mmap_mode=mmap_mode),
        subject_ids=np.load(resolved / "subject_ids.npy", mmap_mode=mmap_mode),
        trial_indices=np.load(resolved / "trial_indices.npy", mmap_mode=mmap_mode),
        window_indices=np.load(resolved / "window_indices.npy", mmap_mode=mmap_mode),
        graph_features=(
            np.load(graph_features_path, mmap_mode=mmap_mode)
            if graph_features_path.is_file()
            else None
        ),
    )


class PlvLiu2024GraphDataset(Dataset[Data]):
    """Create PyG samples lazily from the saved dense graph arrays."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_OUTPUT_DIR,
        *,
        mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.metadata = load_graph_metadata(self.data_dir)
        self.arrays = load_graph_arrays(self.data_dir, mmap_mode=mmap_mode)

    def __len__(self) -> int:
        return len(self.arrays.labels)

    def __getitem__(self, index: int) -> Data:
        components = GraphComponents(
            node_features=np.asarray(self.arrays.node_features[index]),
            plv_matrix=np.asarray(self.arrays.plv_matrices[index]),
            adjacency_matrix=np.asarray(self.arrays.adjacency_matrices[index]),
            graph_features=(
                np.asarray(self.arrays.graph_features[index])
                if self.arrays.graph_features is not None
                else np.empty(0, dtype=np.float32)
            ),
        )
        graph = create_pyg_graph(
            components,
            label=int(self.arrays.labels[index]),
            subject_id=int(self.arrays.subject_ids[index]),
            trial_index=int(self.arrays.trial_indices[index]),
            window_index=int(self.arrays.window_indices[index]),
        )
        validate_pyg_graph(
            graph,
            expected_nodes=int(self.metadata["n_nodes"]),
            expected_node_features=int(self.metadata["n_node_features"]),
            expected_graph_features=int(self.metadata.get("n_graph_features", 0)),
        )
        return graph
