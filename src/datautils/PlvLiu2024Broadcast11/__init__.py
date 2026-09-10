"""Standalone generator for the inferred Liu2024 29-by-11 representation."""

from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, Broadcast11Config
from .features import (
    BROADCAST_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    Broadcast11FeatureResult,
    calculate_broadcast_11_features,
)
from .saved_dataset import PlvLiu2024GraphDataset


def generate_broadcast_11_dataset(
    config: Broadcast11Config | None = None,
    *,
    overwrite: bool = False,
    limit: int | None = None,
) -> Path:
    """Lazily invoke the standalone generator."""

    from .generate_dataset import generate_broadcast_11_dataset as generate

    return generate(config, overwrite=overwrite, limit=limit)


def verify_broadcast_11_dataset(
    data_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Lazily invoke saved-dataset verification."""

    from .verify_dataset import verify_broadcast_11_dataset as verify

    verify(data_dir)


__all__ = [
    "BROADCAST_FEATURE_NAMES",
    "DEFAULT_OUTPUT_DIR",
    "NODE_FEATURE_NAMES",
    "Broadcast11Config",
    "Broadcast11FeatureResult",
    "PlvLiu2024GraphDataset",
    "calculate_broadcast_11_features",
    "generate_broadcast_11_dataset",
    "verify_broadcast_11_dataset",
]
