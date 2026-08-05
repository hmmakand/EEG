"""Tensor dataset wrappers and legacy notebook-only 80/20 helpers.

The controlled comparison entry points use ``protocols.build_full_target_loso_fold``:
source-only never adapts on target signals, while transductive SPPM adapts
without labels on the complete held-out subject and evaluates that same subject
afterward. The chronological adaptation/test builders remain only so the
existing exploratory notebooks continue to import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from eeg_bci.cfspmnet.canonicalization import build_flip_indices, to_affected_unaffected
from eeg_bci.cfspmnet.sppm import compute_private_signature_features
from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.splitting.sources import _chronological_split_indices


def materialize_canonical_trials(
    windows: BraindecodeLikeDataset, metadata: pd.DataFrame, participants: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize every window into canonicalized (x, y) arrays plus subject ids.

    Done once for the whole multi-subject dataset (not per LOSO fold): pulls
    every window out of the lazy `windows` dataset, applies the per-subject
    channel-flip + affected/unaffected label remap, and returns dense arrays
    so downstream fold-building is just numpy masking.
    """

    flip_indices = build_flip_indices()
    n_windows = len(windows)
    subject_values = metadata["subject"].to_numpy()
    raw_targets = metadata["target"].to_numpy()

    x_first, _, *_ = windows[0]
    canonical_x = np.empty((n_windows, *x_first.shape), dtype=np.float32)
    canonical_y = np.empty(n_windows, dtype=np.int64)

    for i in range(n_windows):
        x, _, *_ = windows[i]
        subject_id = int(subject_values[i])
        paralysis_side = str(participants.loc[subject_id, "ParalysisSide"])
        canonical_x[i] = x[flip_indices] if paralysis_side == "left" else x
        canonical_y[i] = to_affected_unaffected(int(raw_targets[i]), paralysis_side)

    return canonical_x, canonical_y, subject_values.astype(np.int64)


class ArrayClassificationDataset(Dataset):
    """Plain in-memory (x, y) classification dataset (source / target-test)."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


class TargetPseudoDataset(Dataset):
    """Unlabeled target-adaptation windows with refreshable pseudo-labels (Sec. 2.3.2)."""

    def __init__(self, x: np.ndarray, signature_vectors: np.ndarray, n_classes: int = 2):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.signature_vectors = torch.as_tensor(signature_vectors, dtype=torch.float32)
        self.pseudo_labels = torch.zeros((len(x), n_classes), dtype=torch.float32)

    def update_pseudo_labels(self, indices, labels) -> None:
        self.pseudo_labels[torch.as_tensor(indices)] = torch.as_tensor(labels)

    def active_ratio(self) -> float:
        return float((self.pseudo_labels.sum(dim=1) > 0).float().mean().item())

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.pseudo_labels[item], self.signature_vectors[item], item


@dataclass(frozen=True)
class CFSPMNetLosoFold:
    held_out_subject: int
    source_dataset: ArrayClassificationDataset
    source_signature_vectors: np.ndarray
    source_labels: np.ndarray
    target_adapt_dataset: TargetPseudoDataset
    target_test_dataset: ArrayClassificationDataset


def chronological_adapt_test_indices(
    windows: BraindecodeLikeDataset, *, test_size: float = 0.2, stratify: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Per-subject leakage-safe chronological split, computed once for all subjects."""

    return _chronological_split_indices(windows, test_size=test_size, stratify=stratify)


def build_loso_fold(
    canonical_x: np.ndarray,
    canonical_y: np.ndarray,
    subject_ids: np.ndarray,
    adapt_indices_all: np.ndarray,
    test_indices_all: np.ndarray,
    held_out_subject: int,
    n_classes: int = 2,
) -> CFSPMNetLosoFold:
    """Build one CFSPMNet LOSO fold from pre-materialized, pre-split arrays."""

    held_out_mask = subject_ids == held_out_subject
    if not held_out_mask.any():
        raise ValueError(f"Subject {held_out_subject} not found in subject_ids.")

    source_mask = ~held_out_mask
    target_adapt_idx = np.intersect1d(np.flatnonzero(held_out_mask), adapt_indices_all)
    target_test_idx = np.intersect1d(np.flatnonzero(held_out_mask), test_indices_all)
    if len(target_adapt_idx) == 0 or len(target_test_idx) == 0:
        raise ValueError(f"Subject {held_out_subject} has an empty adapt/test partition.")

    source_x = canonical_x[source_mask]
    source_y = canonical_y[source_mask]
    source_signature_vectors = compute_private_signature_features(source_x)

    target_adapt_x = canonical_x[target_adapt_idx]
    target_adapt_signature_vectors = compute_private_signature_features(target_adapt_x)

    return CFSPMNetLosoFold(
        held_out_subject=held_out_subject,
        source_dataset=ArrayClassificationDataset(source_x, source_y),
        source_signature_vectors=source_signature_vectors,
        source_labels=source_y,
        target_adapt_dataset=TargetPseudoDataset(
            target_adapt_x, target_adapt_signature_vectors, n_classes=n_classes
        ),
        target_test_dataset=ArrayClassificationDataset(
            canonical_x[target_test_idx], canonical_y[target_test_idx]
        ),
    )
