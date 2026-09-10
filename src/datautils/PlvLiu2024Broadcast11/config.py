"""Configuration for the standalone manuscript broadcast-11 dataset.

This package has no import dependency on ``PlvLiu2024``: the parameter
validation below is an inlined equivalent of
``PlvLiu2024.core.config.validate_config``, kept here instead of importing the
generic ``PlvGraphConfig`` dataclass this package no longer uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "moabb"
    / "Preprocessed-Gamma-31-40-MNE-liu2024-data"
)
DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "data"
    / "moabb"
    / "Graph-Gamma-PLV-Manuscript-31-40-11-liu2024-data"
)


@dataclass(frozen=True)
class Broadcast11Config:
    """Fixed interpretation and paths for generating 11 node features."""

    source_dir: Path = DEFAULT_SOURCE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    expected_channels: int = 29
    expected_samples: int = 2000
    sampling_frequency_hz: float = 500.0
    frequency_band_hz: tuple[float, float] = (31.0, 40.0)
    plv_threshold: float = 0.3
    """Manuscript's ablation-selected graph rule (Table 6): retain and
    binarize PLV edges >= this threshold. Replaces the top-k=4 union rule
    this package previously used."""
    hilbert_edge_trim_seconds: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_dir", Path(self.source_dir).expanduser().resolve()
        )
        object.__setattr__(
            self, "output_dir", Path(self.output_dir).expanduser().resolve()
        )
        if self.source_dir == self.output_dir:
            raise ValueError("source_dir and output_dir must be different")
        if self.expected_channels < 2:
            raise ValueError("expected_channels must be at least two")
        if self.expected_samples < 3:
            raise ValueError("expected_samples must be at least three")
        if self.sampling_frequency_hz <= 0:
            raise ValueError("sampling_frequency_hz must be positive")
        low, high = self.frequency_band_hz
        if not 0 < low < high < self.sampling_frequency_hz / 2:
            raise ValueError("frequency_band_hz must lie inside the Nyquist frequency")
        if not 0 <= self.plv_threshold <= 1:
            raise ValueError("plv_threshold must be between zero and one")
        if self.hilbert_edge_trim_seconds < 0:
            raise ValueError("hilbert_edge_trim_seconds cannot be negative")
        if 2 * self.hilbert_edge_trim_samples >= self.expected_samples:
            raise ValueError("Hilbert boundary trimming removes the complete signal")

    @property
    def hilbert_edge_trim_samples(self) -> int:
        """Return boundary-trim duration converted to signal samples."""

        return round(self.hilbert_edge_trim_seconds * self.sampling_frequency_hz)
