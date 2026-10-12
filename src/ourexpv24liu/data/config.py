"""Every rule that turns a recording into graphs. Changing a value changes the experiment.

Two datasets are supported; `liu2024_config()` and `bci2a_config()` return
their rules. Everything after the reading of the trials is shared.
"""
from dataclasses import dataclass, replace


BCI2A_ALL_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
LIU2024_ALL_SUBJECTS = tuple(range(1, 51))
# Three recordings of the published dataset are copies of another subject's
# recording (same EEG, different patient entry): 5 of 4, 15 of 11, 49 of 47.
# With both members of a pair, leave-one-subject-out trains on the test data.
LIU2024_DUPLICATES = {5: 4, 15: 11, 49: 47}
LIU2024_SUBJECTS = tuple(s for s in LIU2024_ALL_SUBJECTS if s not in LIU2024_DUPLICATES)
DATASETS = ("liu2024", "bci2a")


@dataclass(frozen=True)
class DataConfig:
    # ---- which trials -------------------------------------------------------
    dataset: str = "liu2024"
    subjects: tuple[int, ...] = LIU2024_SUBJECTS
    num_channels: int = 29              # EEG electrodes, in the order of the reader
    sample_rate: int = 250              # of the trials handed to the pipeline
    trial_seconds: float = 8.0          # length of the stretch that is read and filtered
    # bci2a only. None reads every session that contains trials.
    session_indices: tuple[int, ...] | None = None

    # ---- signal -------------------------------------------------------------
    band: tuple[float, float] = (8.0, 30.0)   # zero-phase Butterworth band-pass
    filter_order: int = 5
    # Analysed part, in seconds from the trial's own motor-imagery onset.
    window_seconds: tuple[float, float] = (0.0, 4.0)

    # ---- node features: band power per electrode ----------------------------
    feature_band_edges: tuple[float, ...] = (8, 12, 16, 20, 24, 28, 32)
    welch_seconds: float = 2.0
    welch_overlap: float = 0.5

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}")
        if not self.subjects or len(set(self.subjects)) != len(self.subjects):
            raise ValueError("subjects must be a non-empty list without repeats")
        if not self.window_seconds[0] < self.window_seconds[1]:
            raise ValueError("window_seconds must be (start, stop) with start < stop")

    @property
    def trial_samples(self) -> int:
        return int(self.trial_seconds * self.sample_rate)

    @property
    def window_offsets(self) -> tuple[int, int]:
        """Window start and stop in samples, relative to the motor-imagery onset."""
        start, stop = (int(round(second * self.sample_rate)) for second in self.window_seconds)
        return start, stop

    @property
    def window_samples(self) -> int:
        start, stop = self.window_offsets
        return stop - start

    @property
    def feature_bands(self) -> tuple[tuple[float, float], ...]:
        return tuple(zip(self.feature_band_edges[:-1], self.feature_band_edges[1:]))


def liu2024_config(all_subjects: bool = False, **overrides) -> DataConfig:
    """Liu et al. 2024: acute stroke patients, 29 electrodes, 40 trials each.

    A trial is one 8 s epoch of the published recording. The imagery lasts
    4 s; all of it is analysed. The 47 distinct recordings are used unless
    `all_subjects` is set, which adds the three copies (see LIU2024_DUPLICATES).
    """
    subjects = LIU2024_ALL_SUBJECTS if all_subjects else LIU2024_SUBJECTS
    return replace(DataConfig(subjects=subjects), **overrides)


def bci2a_config(all_subjects: bool = False, **overrides) -> DataConfig:
    """BCI Competition IV 2a, as in version 23: 22 electrodes, 144 left/right trials each.

    A trial is read from 2 s before the cue to 2 s after it; the 2 s after the
    cue are analysed. Subject 4 is left out unless `all_subjects` is set.
    """
    subjects = BCI2A_ALL_SUBJECTS if all_subjects else (1, 2, 3, 5, 6, 7, 8, 9)
    return replace(DataConfig(dataset="bci2a", subjects=subjects, num_channels=22, trial_seconds=4.0,
                              window_seconds=(0.0, 2.0)), **overrides)


def dataset_config(dataset: str, **overrides) -> DataConfig:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    return liu2024_config(**overrides) if dataset == "liu2024" else bci2a_config(**overrides)
