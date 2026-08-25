"""Configuration for Liu2024 experiment data loaders."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoaderConfig:
    """Runtime options shared by every evaluation protocol."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers > 0")

