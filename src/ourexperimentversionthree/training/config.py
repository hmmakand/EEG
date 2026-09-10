"""Configuration for the dedicated broadcast-11 LOSO experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SeedStrategy = Literal["shared", "per_fold"]


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and runtime options shared by all LOSO folds.

    Version three has no dataset selector: it always trains on the canonical
    manuscript broadcast-11 dataset exposed by this experiment's data module.
    """

    run_name: str | None = None
    overwrite: bool = False
    seed_strategy: SeedStrategy = "shared"
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.005
    """Manuscript's tuned value for gamma-band + PLV (Table 2)."""
    weight_decay: float = 1e-4
    patience: int = 10
    minimum_improvement: float = 1e-4
    gradient_clip_norm: float | None = 1.0
    validation_subjects: int = 5
    num_workers: int = 0
    seed: int = 42
    device: str | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    save_outputs: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        if self.run_name is not None:
            if not self.run_name or self.run_name in {".", ".."}:
                raise ValueError("run_name must be a non-empty directory name")
            if Path(self.run_name).name != self.run_name:
                raise ValueError("run_name cannot contain path separators")
        if self.seed_strategy not in ("shared", "per_fold"):
            raise ValueError("seed_strategy must be 'shared' or 'per_fold'")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if (
            not math.isfinite(self.minimum_improvement)
            or self.minimum_improvement < 0
        ):
            raise ValueError(
                "minimum_improvement must be finite and non-negative"
            )
        if self.gradient_clip_norm is not None and (
            not math.isfinite(self.gradient_clip_norm)
            or self.gradient_clip_norm <= 0
        ):
            raise ValueError(
                "gradient_clip_norm must be finite and positive or None"
            )
        if self.validation_subjects <= 0:
            raise ValueError("validation_subjects must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be in the interval [0, 2**32 - 1]")
        if self.device not in (None, "cpu", "cuda"):
            raise ValueError("device must be None, 'cpu', or 'cuda'")
