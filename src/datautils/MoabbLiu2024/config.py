"""Configuration shared by the Liu2024 preparation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "moabb"
DEFAULT_L_FREQ = 8.0
DEFAULT_H_FREQ = 32.0
DEFAULT_SFREQ = 500.0
DEFAULT_STANDARDIZE_INIT_SECONDS = 4.0


@dataclass(frozen=True)
class Liu2024Config:
    """Parameters for EEG preprocessing and motor-imagery window creation."""

    data_root: Path = DEFAULT_DATA_ROOT
    subject_ids: tuple[int, ...] = field(default_factory=lambda: tuple(range(1, 51)))
    l_freq: float = DEFAULT_L_FREQ
    h_freq: float = DEFAULT_H_FREQ
    target_sfreq: float = DEFAULT_SFREQ
    trial_start_offset_seconds: float = 0.0
    trial_stop_offset_seconds: float = 0.0
    standardize: bool = True
    standardize_factor_new: float = 1e-3
    standardize_init_block_seconds: float = DEFAULT_STANDARDIZE_INIT_SECONDS
    preload_windows: bool = True
    n_jobs: int = 1
    event_mapping: dict[str, int] = field(
        default_factory=lambda: {"left_hand": 0, "right_hand": 1}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root).expanduser().resolve())
        if not self.subject_ids:
            raise ValueError("subject_ids cannot be empty")
        if len(self.subject_ids) != len(set(self.subject_ids)):
            raise ValueError("subject_ids contains duplicates")
        if any(subject < 1 or subject > 50 for subject in self.subject_ids):
            raise ValueError("Liu2024 subject IDs must be between 1 and 50")
        if not 0 <= self.l_freq < self.h_freq:
            raise ValueError("Expected 0 <= l_freq < h_freq")
        if self.target_sfreq <= 0:
            raise ValueError("target_sfreq must be positive")
        if self.standardize_init_block_seconds <= 0:
            raise ValueError("standardize_init_block_seconds must be positive")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if set(self.event_mapping) != {"left_hand", "right_hand"}:
            raise ValueError("event_mapping must contain only left_hand and right_hand")

    @property
    def standardize_init_block_size(self) -> int:
        """Initialization duration converted from seconds to signal samples."""

        return round(self.standardize_init_block_seconds * self.target_sfreq)
