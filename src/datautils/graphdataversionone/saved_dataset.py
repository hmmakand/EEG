"""Load saved Liu2024 feature variants and assemble aligned trial graphs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


NODE_FILES = {
    "without_csd": "node_features_without_csd.npy",
    "csd": "node_features_csd.npy",
}
EDGE_FILES = {
    f"{method}_{source}": f"edge_attr_{method}_{source}.npy"
    for method in ("wpli", "plv", "icoh_abs")
    for source in ("without_csd", "csd")
}


@dataclass(frozen=True)
class SavedDataset:
    """Memory-mapped arrays and shared records for the saved feature dataset.

    Parameters
    ----------
    path : pathlib.Path
        Directory containing the feature arrays and metadata.
    nodes : mapping of str to numpy.ndarray
        ``without_csd`` and ``csd`` arrays of shape ``(N, 29, 6)``.
    edges : mapping of str to numpy.ndarray
        Six connectivity arrays, each of shape ``(N, 812, 5)``.
    edge_index : numpy.ndarray
        Shared directed electrode pairs, shape ``(2, 812)``.
    labels : numpy.ndarray
        Integer imagery labels, shape ``(N,)``.
    samples : list of dict
        Trial records in the same order as the first array dimension.
    metadata : dict
        Saved feature names, bands, channel order, and calculation settings.

    Notes
    -----
    Arrays opened by :func:`load_dataset` are read-only memory maps. Graph
    extraction copies the selected arrays, so experiments cannot modify files.
    No normalization or connectivity recalculation is performed.
    """

    path: Path
    nodes: Mapping[str, np.ndarray]
    edges: Mapping[str, np.ndarray]
    edge_index: np.ndarray
    labels: np.ndarray
    samples: list[dict[str, Any]]
    metadata: dict[str, Any]

    def __len__(self) -> int:
        """Return the number of aligned trials without reading feature data.

        Returns
        -------
        int
            Length of the saved label array.
        """
        return len(self.labels)

    def get_graph(
        self,
        index: int,
        node_variant: str = "without_csd",
        edge_variants: Sequence[str] = ("wpli_without_csd",),
        *,
        as_pyg: bool = False,
    ) -> dict[str, Any] | Any:
        """Assemble one trial using independently selected node/edge variants.

        Parameters
        ----------
        index : int
            Zero-based trial index. Negative indices are rejected.
        node_variant : {"without_csd", "csd"}
            Source for the six node features, retaining metadata units.
        edge_variants : sequence of str
            Nonempty sequence of unique connectivity variant names. Columns
            are concatenated in this sequence, then in each variant's saved
            band order. The returned names describe every output column.
        as_pyg : bool
            Return ``torch_geometric.data.Data`` instead of a NumPy dictionary.
            PyTorch and PyG are imported only when this option is requested.

        Returns
        -------
        dict or torch_geometric.data.Data
            ``x`` has shape ``(29, 6)``, ``edge_index`` has shape ``(2, 812)``,
            and ``edge_attr`` has shape ``(812, 5 * len(edge_variants))``.
            Connectivity is dimensionless. The dictionary additionally has
            an integer ``y``, trial record, and explicit node/edge column names.

        Raises
        ------
        IndexError
            If the trial index is outside the saved dataset.
        ValueError
            If a variant is unknown, duplicated, or no edges are selected.
        ImportError
            If ``as_pyg=True`` and the optional dependencies are unavailable.

        Notes
        -----
        Copies are returned; neither source memory maps nor files are modified.
        Fit any subsequent normalization using training samples only.
        """
        return get_graph(self, index, node_variant, edge_variants, as_pyg=as_pyg)


def _read_samples(path: Path) -> list[dict[str, Any]]:
    """Read ordered TSV records and restore documented numeric columns.

    Parameters
    ----------
    path : pathlib.Path
        Path to ``samples.tsv``.

    Returns
    -------
    list of dict
        Records with integer indices/labels and sampling frequency in Hz.

    Raises
    ------
    ValueError
        If a required column is missing or contains an invalid number.
    """
    integer_fields = (
        "sample_index", "subject", "trial_index", "label", "start_sample",
        "stop_sample",
    )
    required = set(integer_fields) | {"target", "source_file", "sampling_frequency_hz"}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing sample columns {sorted(missing)}")
        records = []
        for row_number, row in enumerate(reader, start=2):
            try:
                for field in integer_fields:
                    row[field] = int(row[field])
                row["sampling_frequency_hz"] = float(row["sampling_frequency_hz"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid numeric value on row {row_number}") from exc
            records.append(row)
    return records


def load_dataset(path: str | Path) -> SavedDataset:
    """Open all saved feature arrays as read-only NumPy memory maps.

    Parameters
    ----------
    path : str or pathlib.Path
        Dataset directory containing the files specified by ``DATASET_PLAN.md``.

    Returns
    -------
    SavedDataset
        Arrays, typed sample records, and parsed metadata. Only metadata and
        trial records are read eagerly; feature values remain memory mapped.

    Raises
    ------
    ValueError
        If required metadata, shapes, dtypes, or record counts disagree.
    FileNotFoundError
        If the dataset is incomplete.

    Notes
    -----
    This checks the file schema. Use ``validate_saved_dataset`` for full
    finite-value, connectivity, label, and subject-count validation. It writes
    no files and never unpickles NumPy object arrays.
    """
    path = Path(path).expanduser().resolve()
    with (path / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    required = (
        "n_samples", "channel_names", "node_feature_names", "band_names",
        "class_mapping", "edge_variants", "arrays",
    )
    missing = set(required) - metadata.keys()
    if missing:
        raise ValueError(f"{path}: metadata is missing {sorted(missing)}")
    n_samples = metadata["n_samples"]
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("metadata n_samples must be a positive integer")
    for field, expected in (("channel_names", 29), ("node_feature_names", 6), ("band_names", 5)):
        names = metadata[field]
        if not isinstance(names, list) or len(names) != expected or len(set(names)) != expected:
            raise ValueError(f"metadata {field} must contain {expected} distinct names")
    if metadata["class_mapping"] != {"left_hand": 0, "right_hand": 1}:
        raise ValueError("metadata class_mapping must map left_hand to 0 and right_hand to 1")

    expected_shapes = {
        **{filename: (n_samples, 29, 6) for filename in NODE_FILES.values()},
        **{filename: (n_samples, 812, 5) for filename in EDGE_FILES.values()},
        "edge_index.npy": (2, 812),
        "labels.npy": (n_samples,),
    }
    arrays = {}
    for filename, expected_shape in expected_shapes.items():
        description = metadata["arrays"].get(filename)
        if not isinstance(description, dict) or not {"shape", "dtype"} <= description.keys():
            raise ValueError(f"metadata arrays is missing shape/dtype for {filename}")
        array = np.load(path / filename, mmap_mode="r", allow_pickle=False)
        if array.shape != expected_shape or list(array.shape) != description["shape"]:
            raise ValueError(
                f"{filename}: shape {array.shape} disagrees with expected {expected_shape} or metadata"
            )
        if array.dtype != np.dtype(description["dtype"]):
            raise ValueError(f"{filename}: dtype {array.dtype} disagrees with metadata")
        if filename in ("labels.npy", "edge_index.npy"):
            if not np.issubdtype(array.dtype, np.integer):
                raise ValueError(f"{filename}: expected integer dtype, got {array.dtype}")
        elif not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{filename}: expected floating-point features, got {array.dtype}")
        arrays[filename] = array

    for variant, filename in EDGE_FILES.items():
        entry = metadata["edge_variants"].get(variant)
        if not isinstance(entry, dict) or entry.get("file") != filename:
            raise ValueError(f"metadata edge variant {variant!r} must reference {filename}")
        if entry.get("band_names") != metadata["band_names"]:
            raise ValueError(f"metadata edge variant {variant!r} has a different band order")
    samples = _read_samples(path / "samples.tsv")
    if len(samples) != n_samples:
        raise ValueError(f"samples.tsv has {len(samples)} rows; expected {n_samples}")
    return SavedDataset(
        path=path,
        nodes={name: arrays[filename] for name, filename in NODE_FILES.items()},
        edges={name: arrays[filename] for name, filename in EDGE_FILES.items()},
        edge_index=arrays["edge_index.npy"],
        labels=arrays["labels.npy"],
        samples=samples,
        metadata=metadata,
    )


def get_graph(
    dataset: SavedDataset,
    index: int,
    node_variant: str = "without_csd",
    edge_variants: Sequence[str] = ("wpli_without_csd",),
    *,
    as_pyg: bool = False,
) -> dict[str, Any] | Any:
    """Select a trial and concatenate named connectivity features in order.

    Parameters
    ----------
    dataset : SavedDataset
        Dataset returned by :func:`load_dataset`.
    index : int
        Zero-based sample index; negative indices are rejected.
    node_variant : {"without_csd", "csd"}
        Source for the six electrode features.
    edge_variants : sequence of str
        Unique edge variants, in desired output column-group order.
    as_pyg : bool
        Return a PyG ``Data`` object using optional imports if true.

    Returns
    -------
    dict or torch_geometric.data.Data
        Independent graph arrays/tensors and corresponding feature names.
        Node units follow metadata; all connectivity values are dimensionless.

    Raises
    ------
    IndexError
        If ``index`` is invalid.
    ValueError
        If variant selection is empty, duplicated, or unknown.
    ImportError
        If requested PyG dependencies are missing.

    Notes
    -----
    No input arrays are modified. See :meth:`SavedDataset.get_graph` for shapes.
    """
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(dataset):
        raise IndexError(f"Trial index must be an integer in [0, {len(dataset)}); got {index!r}")
    if node_variant not in dataset.nodes:
        raise ValueError(f"Unknown node variant {node_variant!r}; choose {list(dataset.nodes)}")
    if isinstance(edge_variants, str):
        raise ValueError("edge_variants must be a sequence of names, not a single string")
    selected = tuple(edge_variants)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one edge variant, without duplicates")
    unknown = set(selected) - dataset.edges.keys()
    if unknown:
        raise ValueError(f"Unknown edge variants {sorted(unknown)}; choose {list(dataset.edges)}")

    graph = {
        "x": np.array(dataset.nodes[node_variant][index], copy=True),
        "edge_index": np.array(dataset.edge_index, copy=True),
        "edge_attr": np.concatenate([dataset.edges[name][index] for name in selected], axis=-1),
        "y": int(dataset.labels[index]),
        "sample": dict(dataset.samples[index]),
        "node_variant": node_variant,
        "edge_variants": selected,
        "node_feature_names": list(dataset.metadata["node_feature_names"]),
        "edge_feature_names": [
            f"{name}:{band}"
            for name in selected
            for band in dataset.metadata["edge_variants"][name]["band_names"]
        ],
    }
    if not as_pyg:
        return graph
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise ImportError("as_pyg=True requires torch and torch-geometric") from exc
    return Data(
        x=torch.from_numpy(graph["x"]),
        edge_index=torch.from_numpy(graph["edge_index"]).long(),
        edge_attr=torch.from_numpy(graph["edge_attr"]),
        y=torch.tensor([graph["y"]], dtype=torch.long),
        sample=graph["sample"],
        node_variant=node_variant,
        edge_variants=list(selected),
        node_feature_names=graph["node_feature_names"],
        edge_feature_names=graph["edge_feature_names"],
    )
