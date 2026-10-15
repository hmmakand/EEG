"""Every rule that turns a recording into graphs. Changing a value changes the experiment.

Three datasets are supported; `liu2024_config()`, `jia2019_config()` and
`bci2a_config()` return their rules. Everything after the reading of the trials is shared.
"""
from dataclasses import dataclass, replace


BCI2A_ALL_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
LIU2024_ALL_SUBJECTS = tuple(range(1, 51))
# Three recordings of the published dataset are copies of another subject's
# recording (same EEG, different patient entry): 5 of 4, 15 of 11, 49 of 47.
# With both members of a pair, leave-one-subject-out trains on the test data.
LIU2024_DUPLICATES = {5: 4, 15: 11, 49: 47}
LIU2024_SUBJECTS = tuple(s for s in LIU2024_ALL_SUBJECTS if s not in LIU2024_DUPLICATES)
# Imagery against rest: the imagery of subject 6 starts late in almost every trial, so that the break
# is cut off by the end of the epoch; one of its 40 trials is usable.
LIU2024_IMAGERY_REST_EXCLUDED = (6,)
JIA2019_SUBJECTS = tuple(range(1, 16))
DATASETS = ("liu2024", "jia2019", "bci2a")
TASKS = ("left_right", "imagery_rest")
# imagery_rest: seconds from the trial's imagery onset at which the two samples start, and their length.
# The imagery lasts 4 s and the break 2 s; the first second of each is left out, and the length is
# what remains of the break inside the 8 s epoch for a trial with the usual timing.
IMAGERY_REST = dict(imagery_start=1.0, rest_start=4.0, seconds=1.9)
JIA2019_TRIAL_SECONDS = 6.832        # 1,708 samples at 250 Hz; the shortest published epoch is 3,500 samples at 512 Hz
JIA2019_WINDOW = (0.0, 3.5)          # from the cue, 3 s into the epoch, almost to the end of the epoch


@dataclass(frozen=True)
class DataConfig:
    # ---- which trials -------------------------------------------------------
    dataset: str = "liu2024"
    subjects: tuple[int, ...] = LIU2024_SUBJECTS
    num_channels: int = 29              # EEG electrodes, in the order of the reader
    sample_rate: int = 250              # of the trials handed to the pipeline
    trial_seconds: float = 8.0          # length of the stretch that is read and filtered
    # What is classified:
    #   "left_right"    left-hand against right-hand imagery (label 0 / 1): every dataset
    #   "imagery_rest"  liu2024 only, and the default of liu2024_config(): a stretch of the imagery
    #                   (label 1) against a stretch of the break that follows it (label 0), two
    #                   samples per trial; see IMAGERY_REST
    task: str = "left_right"
    # bci2a only. None reads every session that contains trials.
    session_indices: tuple[int, ...] | None = None

    # ---- signal -------------------------------------------------------------
    band: tuple[float, float] = (8.0, 30.0)   # zero-phase Butterworth band-pass
    filter_order: int = 5
    # Analysed part, in seconds from the trial's own motor-imagery onset.
    window_seconds: tuple[float, float] = (0.0, 4.0)

    # ---- optional features (off by default) ---------------------------------
    # Tangent-space covariance: the matrix logarithm of the trial's covariance. Its diagonal becomes
    # a seventh node feature and its off-diagonal entries a third edge feature.
    tangent_features: bool = False
    # Low-frequency potentials: the trial band-passed to this band (for example (0.5, 4.0) Hz),
    # aligned per subject in that band and stored at low_sample_rate as a second signal.
    low_band: tuple[float, float] | None = None
    low_filter_order: int = 3
    low_sample_rate: int = 25

    # ---- node features: band power per electrode ----------------------------
    feature_band_edges: tuple[float, ...] = (8, 12, 16, 20, 24, 28, 32)
    welch_seconds: float = 2.0
    welch_overlap: float = 0.5

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}")
        if self.task not in TASKS or (self.task == "imagery_rest" and self.dataset != "liu2024"):
            raise ValueError(f"task must be one of {TASKS}; imagery_rest is defined for liu2024 only")
        if not self.subjects or len(set(self.subjects)) != len(self.subjects):
            raise ValueError("subjects must be a non-empty list without repeats")
        if self.low_band is not None and (not 0 < self.low_band[0] < self.low_band[1]
                                          or self.sample_rate % self.low_sample_rate
                                          or self.low_band[1] >= self.low_sample_rate / 2):
            raise ValueError("low_band needs 0 < low < high < low_sample_rate / 2, and a low_sample_rate that "
                             "divides sample_rate")
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
    def low_samples(self) -> int:
        """Samples of the low-frequency signal of a trial; 0 without a low band."""
        if self.low_band is None:
            return 0
        return len(range(0, self.window_samples, self.sample_rate // self.low_sample_rate))

    @property
    def feature_bands(self) -> tuple[tuple[float, float], ...]:
        return tuple(zip(self.feature_band_edges[:-1], self.feature_band_edges[1:]))


def liu2024_config(all_subjects: bool = False, **overrides) -> DataConfig:
    """Liu et al. 2024: acute stroke patients, 29 electrodes, 40 trials each.

    The default task is imagery against rest (`task="imagery_rest"`), the
    experiment with the best result on this dataset: two samples of 1.9 s per
    trial, 46 subjects (the three copies and subject 6 are left out).

    `task="left_right"` gives the experiment of version 24: a trial is one 8 s
    epoch, the 4 s of imagery are analysed, 47 subjects.

    `all_subjects` adds the subjects that are left out by default.
    """
    overrides = {"task": "imagery_rest", **overrides}
    subjects = LIU2024_ALL_SUBJECTS if all_subjects else LIU2024_SUBJECTS
    if overrides["task"] == "imagery_rest":
        overrides = {"window_seconds": (0.0, IMAGERY_REST["seconds"]), **overrides}
        if not all_subjects:
            subjects = tuple(s for s in subjects if s not in LIU2024_IMAGERY_REST_EXCLUDED)
    return replace(DataConfig(subjects=subjects), **overrides)


def bci2a_config(all_subjects: bool = False, **overrides) -> DataConfig:
    """BCI Competition IV 2a, as in version 23: 22 electrodes, 144 left/right trials each.

    A trial is read from 2 s before the cue to 2 s after it; the 2 s after the
    cue are analysed. Subject 4 is left out unless `all_subjects` is set.
    """
    subjects = BCI2A_ALL_SUBJECTS if all_subjects else (1, 2, 3, 5, 6, 7, 8, 9)
    return replace(DataConfig(dataset="bci2a", subjects=subjects, num_channels=22, trial_seconds=4.0,
                              window_seconds=(0.0, 2.0)), **overrides)


def jia2019_config(**overrides) -> DataConfig:
    """Jia 2019: 15 stroke patients, 63 electrodes, 80 trials each (40 per class).

    A trial is one published epoch, 6.8 s at 512 Hz, resampled to 250 Hz. The
    cue comes 3 s into the epoch (see data/jia2019.py); the 3.5 s after it are
    analysed.
    """
    return replace(DataConfig(dataset="jia2019", subjects=JIA2019_SUBJECTS, num_channels=63,
                              trial_seconds=JIA2019_TRIAL_SECONDS, window_seconds=JIA2019_WINDOW), **overrides)


def dataset_config(dataset: str, **overrides) -> DataConfig:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    return dict(liu2024=liu2024_config, jia2019=jia2019_config, bci2a=bci2a_config)[dataset](**overrides)
