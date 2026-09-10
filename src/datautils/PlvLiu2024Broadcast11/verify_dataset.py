"""Verify the saved manuscript broadcast-11 dataset and its exact broadcast rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import DEFAULT_OUTPUT_DIR
from .features import BROADCAST_FEATURE_NAMES, NODE_FEATURE_NAMES
from .generate_dataset import DATASET_VARIANT, GRAPH_FORMAT_VERSION
from .saved_dataset import PlvLiu2024GraphDataset


def verify_broadcast_11_dataset(data_dir: str | Path = DEFAULT_OUTPUT_DIR) -> None:
    """Raise unless files, metadata, values, and broadcast semantics are valid."""

    resolved = Path(data_dir).expanduser().resolve()
    metadata = json.loads((resolved / "metadata.json").read_text(encoding="utf-8"))
    expected_metadata = {
        "graph_format_version": GRAPH_FORMAT_VERSION,
        "dataset_variant": DATASET_VARIANT,
        "n_node_features": len(NODE_FEATURE_NAMES),
        "n_graph_features": 0,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Unexpected {key}: {metadata.get(key)!r}")
    if tuple(metadata.get("node_feature_names", ())) != NODE_FEATURE_NAMES:
        raise ValueError("Unexpected node-feature names or order")

    dataset = PlvLiu2024GraphDataset(resolved)
    node_features = np.asarray(dataset.arrays.node_features)
    source_features = np.load(
        resolved / "broadcast_source_features.npy", mmap_mode="r"
    )
    expected_shape = (len(dataset), int(metadata["n_nodes"]), 11)
    if node_features.shape != expected_shape:
        raise ValueError(
            f"Unexpected node_features shape {node_features.shape}; "
            f"expected {expected_shape}"
        )
    if source_features.shape != (len(dataset), len(BROADCAST_FEATURE_NAMES)):
        raise ValueError("Unexpected broadcast_source_features shape")
    if not np.isfinite(node_features).all() or not np.isfinite(source_features).all():
        raise ValueError("Saved broadcast features contain non-finite values")
    expected_broadcast = np.broadcast_to(
        source_features[:, None, :], node_features[:, :, 4:].shape
    )
    if not np.array_equal(node_features[:, :, 4:], expected_broadcast):
        raise ValueError("Columns 4:11 are not exact per-graph broadcasts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    verify_broadcast_11_dataset(args.input)
    print(f"Verified broadcast-11 dataset at {args.input.resolve()}")


if __name__ == "__main__":
    main()
