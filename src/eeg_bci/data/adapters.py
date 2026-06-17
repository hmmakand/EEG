"""Adapters between Braindecode datasets and PyTorch training loops.

Braindecode dataset items usually include signal data, a target label, and extra
metadata such as crop/window indices. The trainer in this project only needs
`(x, y)` tensor pairs, so this module provides a small adapter that preserves the
Braindecode dataset source while exposing a simple PyTorch Dataset interface.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch.utils.data import Dataset


class BraindecodeLikeDataset(Protocol):
    """Minimal protocol shared by Braindecode datasets used in this project."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]: ...


class TensorDatasetFromBraindecode(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Convert Braindecode samples into `(signal, target)` tensors.

    Extra metadata returned by Braindecode is intentionally ignored here; split
    decisions should happen before this adapter is applied.
    """

    def __init__(self, dataset: BraindecodeLikeDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, *_ = self.dataset[index]
        x_tensor = torch.as_tensor(x, dtype=torch.float32)
        y_tensor = torch.as_tensor(y, dtype=torch.long)
        return x_tensor, y_tensor
