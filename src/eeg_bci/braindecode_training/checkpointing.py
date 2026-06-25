from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_classifier_module(classifier: Any, output_dir: Path, checkpoint_name: str) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.module_.state_dict(), checkpoint_dir / checkpoint_name)
