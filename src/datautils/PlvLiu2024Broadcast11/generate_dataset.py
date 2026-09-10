"""Atomically generate the standalone manuscript broadcast-11 graph dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .config import Broadcast11Config
from .graph import create_connectivity_components
from .source import Liu2024SourceArrays, load_source_arrays, validate_source_arrays
from .features import (
    BETWEENNESS_CONFIGURATION,
    BROADCAST_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    PSD_CONFIGURATION,
    TOPOLOGY_CONFIGURATION,
    calculate_broadcast_11_features,
)


GRAPH_FORMAT_VERSION = 3
DATASET_VARIANT = "manuscript_broadcast_11_v3"


def _metadata(
    config: Broadcast11Config,
    source: Liu2024SourceArrays,
    n_graphs: int,
) -> dict[str, object]:
    source_metadata = source.metadata
    labels = np.asarray(source.y[:n_graphs])
    subjects = np.asarray(source.subject_ids[:n_graphs])
    class_counts = Counter(map(int, labels))
    n_nodes = config.expected_channels
    return {
        "dataset": "Liu2024",
        "graph_format_version": GRAPH_FORMAT_VERSION,
        "dataset_variant": DATASET_VARIANT,
        "feature_set": "manuscript_broadcast_11",
        "feature_status": "inferred_manuscript_interpretation",
        "representation": "NumPy dense graph arrays with lazy PyG conversion",
        "source_dataset_dir": str(config.source_dir),
        "source_format_version": source_metadata.get("format_version"),
        "source_window_shape": [n_nodes, config.expected_samples],
        "source_frequency_band_hz": list(config.frequency_band_hz),
        "sampling_frequency_hz": config.sampling_frequency_hz,
        "n_graphs": n_graphs,
        "n_nodes": n_nodes,
        "n_node_features": len(NODE_FEATURE_NAMES),
        "n_graph_features": 0,
        "channel_names": source_metadata["channel_names"],
        "node_feature_names": list(NODE_FEATURE_NAMES),
        "graph_feature_names": [],
        "broadcast_source_feature_names": list(BROADCAST_FEATURE_NAMES),
        "feature_scopes": {
            "node_features": "node",
            "global_features_broadcast_to_nodes": True,
            "broadcast_columns": list(range(4, 11)),
        },
        "broadcast_rule": {
            "source": "seven whole-graph persistent-homology scalars",
            "operation": "repeat each scalar unchanged for all 29 nodes",
            "manuscript_interpretation": "inferred_not_confirmed_node_specific_mapping",
        },
        "feature_configuration": {
            "std_ddof": 0,
            "psd": asdict(PSD_CONFIGURATION),
            "betweenness": asdict(BETWEENNESS_CONFIGURATION),
            "topology": asdict(TOPOLOGY_CONFIGURATION),
        },
        "class_mapping": source_metadata["class_mapping"],
        "class_counts": {
            str(key): value for key, value in sorted(class_counts.items())
        },
        "subjects": sorted(set(map(int, subjects))),
        "connectivity": "phase_locking_value",
        "hilbert_edge_trim_seconds": config.hilbert_edge_trim_seconds,
        "hilbert_edge_trim_samples": (
            config.hilbert_edge_trim_samples
        ),
        "sparsification": {
            "method": "threshold",
            "threshold": config.plv_threshold,
            "binarized": True,
        },
        "edge_weights": "plv",
        "self_edges": False,
        "arrays": {
            "node_features.npy": {
                "shape": [n_graphs, n_nodes, len(NODE_FEATURE_NAMES)],
                "dtype": "float32",
            },
            "broadcast_source_features.npy": {
                "shape": [n_graphs, len(BROADCAST_FEATURE_NAMES)],
                "dtype": "float32",
                "role": "audit copy; not loaded as a model graph feature",
            },
            "plv_matrices.npy": {
                "shape": [n_graphs, n_nodes, n_nodes],
                "dtype": "float32",
            },
            "adjacency_matrices.npy": {
                "shape": [n_graphs, n_nodes, n_nodes],
                "dtype": "float32",
            },
            "labels.npy": {"shape": [n_graphs], "dtype": "int64"},
            "subject_ids.npy": {"shape": [n_graphs], "dtype": "int64"},
            "trial_indices.npy": {"shape": [n_graphs], "dtype": "int64"},
            "window_indices.npy": {"shape": [n_graphs], "dtype": "int64"},
        },
    }


def generate_broadcast_11_dataset(
    config: Broadcast11Config | None = None,
    *,
    overwrite: bool = False,
    limit: int | None = None,
) -> Path:
    """Generate all arrays in a temporary directory and publish atomically."""

    resolved = config or Broadcast11Config()
    source = load_source_arrays(resolved.source_dir)
    validate_source_arrays(source, resolved)
    total_graphs = len(source.X)
    if limit is not None and not 0 < limit <= total_graphs:
        raise ValueError(f"limit must be between 1 and {total_graphs}")
    n_graphs = total_graphs if limit is None else limit
    output_dir = resolved.output_dir
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_dir}; use --overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )

    n_nodes = resolved.expected_channels
    arrays = {
        "node_features": open_memmap(
            temporary / "node_features.npy", mode="w+", dtype=np.float32,
            shape=(n_graphs, n_nodes, len(NODE_FEATURE_NAMES)),
        ),
        "broadcast_source_features": open_memmap(
            temporary / "broadcast_source_features.npy", mode="w+",
            dtype=np.float32, shape=(n_graphs, len(BROADCAST_FEATURE_NAMES)),
        ),
        "plv_matrices": open_memmap(
            temporary / "plv_matrices.npy", mode="w+", dtype=np.float32,
            shape=(n_graphs, n_nodes, n_nodes),
        ),
        "adjacency_matrices": open_memmap(
            temporary / "adjacency_matrices.npy", mode="w+", dtype=np.float32,
            shape=(n_graphs, n_nodes, n_nodes),
        ),
        "labels": open_memmap(
            temporary / "labels.npy", mode="w+", dtype=np.int64,
            shape=(n_graphs,),
        ),
        "subject_ids": open_memmap(
            temporary / "subject_ids.npy", mode="w+", dtype=np.int64,
            shape=(n_graphs,),
        ),
        "trial_indices": open_memmap(
            temporary / "trial_indices.npy", mode="w+", dtype=np.int64,
            shape=(n_graphs,),
        ),
        "window_indices": open_memmap(
            temporary / "window_indices.npy", mode="w+", dtype=np.int64,
            shape=(n_graphs,),
        ),
    }
    try:
        for index in range(n_graphs):
            signals = np.asarray(source.X[index])
            connectivity = create_connectivity_components(signals, resolved)
            features = calculate_broadcast_11_features(
                signals,
                connectivity.plv_matrix,
                connectivity.adjacency_matrix,
                resolved.sampling_frequency_hz,
            )
            arrays["node_features"][index] = features.node_features
            arrays["broadcast_source_features"][index] = (
                features.broadcast_source_features
            )
            arrays["plv_matrices"][index] = connectivity.plv_matrix
            arrays["adjacency_matrices"][index] = connectivity.adjacency_matrix
            arrays["labels"][index] = source.y[index]
            arrays["subject_ids"][index] = source.subject_ids[index]
            arrays["trial_indices"][index] = source.trial_indices[index]
            arrays["window_indices"][index] = index
            if (index + 1) % 100 == 0 or index + 1 == n_graphs:
                print(f"Processed {index + 1}/{n_graphs} graphs", flush=True)
        for array in arrays.values():
            array.flush()
        if not np.isfinite(arrays["node_features"]).all():
            raise ValueError("Generated node features contain non-finite values")
        expected_broadcast = np.broadcast_to(
            arrays["broadcast_source_features"][:, None, :],
            arrays["node_features"][:, :, 4:].shape,
        )
        if not np.array_equal(arrays["node_features"][:, :, 4:], expected_broadcast):
            raise ValueError("Generated topology columns do not broadcast exactly")
        (temporary / "metadata.json").write_text(
            json.dumps(_metadata(resolved, source, n_graphs), indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
    except Exception:
        for array in arrays.values():
            array.flush()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"Broadcast-11 graph dataset saved to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    defaults = Broadcast11Config()
    config = Broadcast11Config(
        source_dir=args.input or defaults.source_dir,
        output_dir=args.output or defaults.output_dir,
    )
    generate_broadcast_11_dataset(
        config, overwrite=args.overwrite, limit=args.limit
    )


if __name__ == "__main__":
    main()
