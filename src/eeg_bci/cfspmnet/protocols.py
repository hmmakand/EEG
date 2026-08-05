"""Full-target LOSO protocols for the CFSPMNet comparison experiments.

This module keeps the comparison boundary explicit:

* source-only folds never construct a target-adaptation dataset;
* transductive SPPM sees every held-out target trial without its label and is
  evaluated on those same trials after adaptation;
* raw left/right and affected/unaffected+hemisphere-flip canonicalization are
  atomic, named choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from eeg_bci.cfspmnet.canonicalization import (
    LIU2024_EEG_CHANNELS,
    LIU2024_FIGSHARE_EEG_CHANNELS,
    canonicalize_subject_trials,
)
from eeg_bci.cfspmnet.datasets import (
    ArrayClassificationDataset,
    TargetPseudoDataset,
)

RAW_LEFT_RIGHT = "raw_left_right"
AFFECTED_UNAFFECTED_FLIP = "affected_unaffected_flip"
SOURCE_ONLY = "source_only"
SPPM_TRANSDUCTIVE = "sppm_transductive"

CanonicalizationMode = Literal["raw_left_right", "affected_unaffected_flip"]
TrainingProtocol = Literal["source_only", "sppm_transductive"]

CLASS_NAMES_BY_CANONICALIZATION: dict[str, list[str]] = {
    RAW_LEFT_RIGHT: ["left_hand", "right_hand"],
    AFFECTED_UNAFFECTED_FLIP: ["affected", "unaffected"],
}


@dataclass(frozen=True)
class FullTargetLosoFold:
    """One complete held-out-subject fold.

    In the transductive protocol, the adaptation and test datasets contain the
    same signals by design. Only the test dataset carries ground-truth labels.
    """

    held_out_subject: int
    source_dataset: ArrayClassificationDataset
    source_labels: np.ndarray
    target_test_dataset: ArrayClassificationDataset
    source_signature_vectors: np.ndarray | None = None
    target_adapt_dataset: TargetPseudoDataset | None = None


def validate_channel_names(channel_names: tuple[str, ...]) -> None:
    """Reject an unexpected montage/order before fixed-index flipping or SPPM."""

    supported = {LIU2024_EEG_CHANNELS, LIU2024_FIGSHARE_EEG_CHANNELS}
    if channel_names not in supported:
        raise ValueError(
            "Unexpected Liu2024 EEG channel order. Expected either the "
            "29-channel MOABB montage or the 30-channel Figshare raw montage; "
            f"got {channel_names}."
        )


def materialize_window_trials(
    windows: Any,
    metadata: pd.DataFrame,
    channel_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize project-preprocessed Braindecode windows without relabeling."""

    validate_channel_names(channel_names)
    if len(windows) == 0:
        raise ValueError("Cannot materialize an empty window dataset.")
    if len(metadata) != len(windows):
        raise ValueError(
            f"Window/metadata length mismatch: {len(windows)} != {len(metadata)}."
        )
    required_columns = {"subject", "target"}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"Window metadata is missing columns: {sorted(missing)}")

    first_x, first_y, *_ = windows[0]
    first_x = np.asarray(first_x)
    expected_shape = (len(channel_names), first_x.shape[-1])
    if first_x.shape != expected_shape:
        raise ValueError(
            f"Window shape {first_x.shape} does not match montage {expected_shape}."
        )

    x = np.empty((len(windows), *expected_shape), dtype=np.float32)
    y = metadata["target"].to_numpy(dtype=np.int64, copy=True)
    subject_ids = metadata["subject"].to_numpy(dtype=np.int64, copy=True)

    for index in range(len(windows)):
        trial_x, trial_y, *_ = windows[index]
        trial_x = np.asarray(trial_x, dtype=np.float32)
        if trial_x.shape != expected_shape:
            raise ValueError(
                f"Window {index} has shape {trial_x.shape}; expected {expected_shape}."
            )
        if int(trial_y) != int(y[index]):
            raise ValueError(
                f"Window {index} target {trial_y} disagrees with metadata {y[index]}."
            )
        x[index] = trial_x

    _validate_trial_arrays(x, y, subject_ids, channel_names)
    return x, y, subject_ids


def canonicalize_trial_arrays(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    *,
    mode: CanonicalizationMode,
    channel_names: tuple[str, ...],
    participants: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one atomic target/hemisphere convention to all trial arrays."""

    _validate_trial_arrays(x, y, subject_ids, channel_names)
    if mode == RAW_LEFT_RIGHT:
        return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int64)
    if mode != AFFECTED_UNAFFECTED_FLIP:
        raise ValueError(
            f"Unsupported canonicalization {mode!r}; expected "
            f"{RAW_LEFT_RIGHT!r} or {AFFECTED_UNAFFECTED_FLIP!r}."
        )
    if participants is None:
        raise ValueError("participants metadata is required for canonicalization.")

    missing_subjects = sorted(
        set(int(v) for v in np.unique(subject_ids)) - set(participants.index)
    )
    if missing_subjects:
        raise ValueError(
            "participants.tsv is missing subjects required for canonicalization: "
            f"{missing_subjects}"
        )

    canonical_x = np.empty_like(x, dtype=np.float32)
    canonical_y = np.empty_like(y, dtype=np.int64)
    for subject_id in np.unique(subject_ids):
        mask = subject_ids == subject_id
        subject_x, subject_y = canonicalize_subject_trials(
            x[mask],
            y[mask],
            int(subject_id),
            participants,
            channel_names=channel_names,
        )
        canonical_x[mask] = subject_x
        canonical_y[mask] = subject_y
    return canonical_x, canonical_y


def build_full_target_loso_fold(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    *,
    held_out_subject: int,
    protocol: TrainingProtocol,
    signature_vectors: np.ndarray | None = None,
    n_classes: int = 2,
) -> FullTargetLosoFold:
    """Create canonical full-subject LOSO or full-target transductive SPPM."""

    if not (len(x) == len(y) == len(subject_ids)):
        raise ValueError("x, y, and subject_ids must have identical lengths.")
    unique_subjects = np.unique(subject_ids)
    if len(unique_subjects) < 2:
        raise ValueError("LOSO requires at least two unique cohort subjects.")
    if held_out_subject not in unique_subjects:
        raise ValueError(f"Held-out subject {held_out_subject} is not in the cohort.")
    if protocol not in {SOURCE_ONLY, SPPM_TRANSDUCTIVE}:
        raise ValueError(f"Unsupported training protocol: {protocol!r}")

    target_mask = subject_ids == held_out_subject
    source_mask = ~target_mask
    if not target_mask.any() or not source_mask.any():
        raise ValueError("A LOSO fold must have non-empty source and target domains.")

    source_x = np.asarray(x[source_mask], dtype=np.float32)
    source_y = np.asarray(y[source_mask], dtype=np.int64)
    target_x = np.asarray(x[target_mask], dtype=np.float32)
    target_y = np.asarray(y[target_mask], dtype=np.int64)
    missing_classes = sorted(
        set(range(n_classes)) - set(int(v) for v in np.unique(source_y))
    )
    if missing_classes:
        raise ValueError(f"Source domain is missing classes: {missing_classes}")

    target_adapt_dataset: TargetPseudoDataset | None = None
    source_signatures: np.ndarray | None = None
    if protocol == SPPM_TRANSDUCTIVE:
        if signature_vectors is None or len(signature_vectors) != len(x):
            raise ValueError(
                "Transductive SPPM requires one precomputed signature vector per trial."
            )
        source_signatures = np.asarray(signature_vectors[source_mask], dtype=np.float32)
        target_signatures = np.asarray(signature_vectors[target_mask], dtype=np.float32)
        target_adapt_dataset = TargetPseudoDataset(
            target_x, target_signatures, n_classes=n_classes
        )
    elif signature_vectors is not None:
        raise ValueError("Source-only folds must not receive target signature vectors.")

    return FullTargetLosoFold(
        held_out_subject=held_out_subject,
        source_dataset=ArrayClassificationDataset(source_x, source_y),
        source_labels=source_y,
        target_test_dataset=ArrayClassificationDataset(target_x, target_y),
        source_signature_vectors=source_signatures,
        target_adapt_dataset=target_adapt_dataset,
    )


def _validate_trial_arrays(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    channel_names: tuple[str, ...],
) -> None:
    validate_channel_names(channel_names)
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (N, C, T), got {x.shape}.")
    if y.ndim != 1 or subject_ids.ndim != 1:
        raise ValueError("y and subject_ids must be one-dimensional.")
    if not (len(x) == len(y) == len(subject_ids)):
        raise ValueError("x, y, and subject_ids must have identical lengths.")
    if x.shape[1] != len(channel_names):
        raise ValueError(
            f"x contains {x.shape[1]} channels but channel_names has "
            f"{len(channel_names)} entries."
        )
    if not np.isfinite(x).all():
        raise ValueError("Trial arrays contain NaN or infinite values.")
    invalid_targets = sorted(set(int(v) for v in np.unique(y)) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Expected binary targets 0/1, got {invalid_targets}.")
