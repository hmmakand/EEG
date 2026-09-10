"""Load and validate the saved Liu2024 EEG source dataset.

The loader memory-maps EEG and annotation arrays from the preprocessed source
directory. Validation guarantees the expected 2,000 windows, 29-channel order,
class balance, subject/trial alignment, and preprocessing metadata before graph
construction begins. This module does not calculate graph features or edges.

This is a self-contained copy of ``src.datautils.PlvLiu2024.source`` kept so
that ``PlvLiu2024Broadcast11`` has no import dependency on ``PlvLiu2024``.
Keep in sync by hand if the original is ever fixed. Validation takes
``Broadcast11Config`` directly instead of the generic ``PlvGraphConfig``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from .config import Broadcast11Config


REQUIRED_SOURCE_FILES = (
    "X.npy",
    "y.npy",
    "subject_ids.npy",
    "trial_indices.npy",
    "metadata.json",
)


@dataclass(frozen=True)
class Liu2024SourceArrays:
    """Memory-mapped source arrays and their JSON metadata."""

    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    trial_indices: np.ndarray
    metadata: dict[str, Any]


def validate_required_source_files(data_dir: Path) -> None:
    """Raise when a required preprocessed source file is missing."""

    missing = [name for name in REQUIRED_SOURCE_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete source dataset at {data_dir}; missing: {missing}")


def load_source_metadata(data_dir: Path) -> dict[str, Any]:
    """Read source metadata from JSON."""

    with (data_dir / "metadata.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def load_source_arrays(
    data_dir: str | Path,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
) -> Liu2024SourceArrays:
    """Load source arrays without copying the complete EEG dataset into memory."""

    resolved = Path(data_dir).expanduser().resolve()
    validate_required_source_files(resolved)
    return Liu2024SourceArrays(
        X=np.load(resolved / "X.npy", mmap_mode=mmap_mode),
        y=np.load(resolved / "y.npy", mmap_mode=mmap_mode),
        subject_ids=np.load(resolved / "subject_ids.npy", mmap_mode=mmap_mode),
        trial_indices=np.load(resolved / "trial_indices.npy", mmap_mode=mmap_mode),
        metadata=load_source_metadata(resolved),
    )


def validate_source_metadata(
    metadata: dict[str, Any], config: "Broadcast11Config"
) -> None:
    """Verify source preprocessing and dimensions against graph configuration."""

    if metadata.get("dataset") != "Liu2024":
        raise ValueError("Expected Liu2024 source metadata")
    if tuple(metadata.get("window_shape", ())) != (
        config.expected_channels,
        config.expected_samples,
    ):
        raise ValueError("Source window_shape does not match graph configuration")
    if float(metadata.get("sampling_frequency_hz", 0.0)) != config.sampling_frequency_hz:
        raise ValueError("Source sampling frequency does not match graph configuration")
    preprocessing = metadata.get("preprocessing", {})
    if tuple(preprocessing.get("bandpass_hz", ())) != config.frequency_band_hz:
        raise ValueError("Source band-pass does not match graph configuration")
    channels = metadata.get("channel_names", [])
    if len(channels) != config.expected_channels or len(set(channels)) != len(channels):
        raise ValueError("Source channel names are missing, duplicated, or incomplete")
    if metadata.get("class_mapping") != {"left_hand": 0, "right_hand": 1}:
        raise ValueError("Unexpected source class mapping")


def validate_source_arrays(
    source: Liu2024SourceArrays, config: "Broadcast11Config"
) -> None:
    """Verify source array shapes, values, subjects, trials, and class balance."""

    validate_source_metadata(source.metadata, config)
    expected_windows = int(source.metadata.get("n_windows", 0))
    expected_x_shape = (
        expected_windows,
        config.expected_channels,
        config.expected_samples,
    )
    if source.X.shape != expected_x_shape:
        raise ValueError(f"Expected X shape {expected_x_shape}, got {source.X.shape}")
    for name, array in (
        ("y", source.y),
        ("subject_ids", source.subject_ids),
        ("trial_indices", source.trial_indices),
    ):
        if array.shape != (expected_windows,):
            raise ValueError(f"Expected {name} shape {(expected_windows,)}, got {array.shape}")
    if not np.isfinite(source.X).all():
        raise ValueError("Source EEG contains non-finite values")
    if Counter(map(int, source.y)) != Counter({0: 1000, 1: 1000}):
        raise ValueError("Expected 1,000 source windows per class")
    if set(map(int, source.subject_ids)) != set(range(1, 51)):
        raise ValueError("Expected source subjects 1 through 50")
    for subject in range(1, 51):
        mask = np.asarray(source.subject_ids) == subject
        if int(mask.sum()) != 40:
            raise ValueError(f"Subject {subject} does not have 40 windows")
        if Counter(map(int, np.asarray(source.y)[mask])) != Counter({0: 20, 1: 20}):
            raise ValueError(f"Subject {subject} does not have 20 windows per class")
        trials = np.asarray(source.trial_indices)[mask]
        if set(map(int, trials)) != set(range(40)):
            raise ValueError(f"Subject {subject} trial indices are not 0 through 39")
