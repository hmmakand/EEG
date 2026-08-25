"""PyTorch dataset for the saved, preprocessed MOABB Liu2024 arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_DATA_ROOT


DEFAULT_PREPROCESSED_DIR = DEFAULT_DATA_ROOT / "Preprocessed-MNE-liu2024-data"
REQUIRED_FILES = (
    "X.npy",
    "y.npy",
    "subject_ids.npy",
    "trial_indices.npy",
    "metadata.json",
)


class Liu2024TorchDataset(Dataset):
    """Access saved Liu2024 windows using memory-mapped NumPy arrays."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_PREPROCESSED_DIR,
        *,
        mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
    ) -> None:
        self.data_dir = Path(data_dir).expanduser().resolve()
        missing = [name for name in REQUIRED_FILES if not (self.data_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete preprocessed Liu2024 dataset at {self.data_dir}; "
                f"missing: {missing}"
            )

        with (self.data_dir / "metadata.json").open(encoding="utf-8") as stream:
            self.metadata: dict[str, Any] = json.load(stream)

        self.X = np.load(self.data_dir / "X.npy", mmap_mode=mmap_mode)
        self.y = np.load(self.data_dir / "y.npy", mmap_mode=mmap_mode)
        self.subject_ids = np.load(
            self.data_dir / "subject_ids.npy", mmap_mode=mmap_mode
        )
        self.trial_indices = np.load(
            self.data_dir / "trial_indices.npy", mmap_mode=mmap_mode
        )
        self._validate()

    def _validate(self) -> None:
        n_windows = len(self.X)
        arrays = (self.y, self.subject_ids, self.trial_indices)
        if any(len(array) != n_windows for array in arrays):
            raise ValueError("Saved Liu2024 arrays have inconsistent lengths")
        if self.X.ndim != 3:
            raise ValueError(f"Expected X with 3 dimensions; got {self.X.shape}")
        if tuple(self.metadata.get("window_shape", ())) != tuple(self.X.shape[1:]):
            raise ValueError("X shape does not match metadata window_shape")
        if self.metadata.get("n_windows") != n_windows:
            raise ValueError("Array length does not match metadata n_windows")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        # A single-window copy avoids exposing a read-only memory map as a
        # writable tensor while keeping the complete dataset off heap.
        x = torch.from_numpy(np.array(self.X[index], dtype=np.float32, copy=True))
        target = int(self.y[index])
        sample_info = {
            "subject": int(self.subject_ids[index]),
            "trial": int(self.trial_indices[index]),
        }
        return x, target, sample_info
