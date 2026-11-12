"""Every rule that turns a recording into graphs. Changing a value changes the experiment.

Two datasets, each with its task:

    liu2024   acute stroke patients: imagery (1) against rest (0)       liu2024_config()
    bci2a     healthy subjects: left hand (0) against right hand (1)    bci2a_config()
    jia2019   stroke patients: left hand (0) against right hand (1)     jia2019_config()
"""
from dataclasses import dataclass, replace

DATASETS = ("liu2024", "bci2a", "jia2019")

# ---- Liu et al. 2024 ----------------------------------------------------------
LIU2024_ALL_SUBJECTS = tuple(range(1, 51))
# Three recordings of the published dataset are copies of another subject's
# recording (same EEG, different patient entry): 5 of 4, 15 of 11, 49 of 47.
# With both members of a pair, leave-one-subject-out trains on the test data.
LIU2024_DUPLICATES = {5: 4, 15: 11, 49: 47}
# The imagery of subject 6 starts late in almost every trial, so that the break
# is cut off by the end of the epoch; one of its 40 trials is usable.
LIU2024_TOO_FEW_TRIALS = (6,)
LIU2024_SUBJECTS = tuple(s for s in LIU2024_ALL_SUBJECTS
                         if s not in LIU2024_DUPLICATES and s not in LIU2024_TOO_FEW_TRIALS)   # 46 patients

# ---- BCI Competition IV 2a ----------------------------------------------------
# All nine subjects. Versions 7 to 23 left subject 4 out (its file has fewer sessions);
# the reader skips the sessions without trials, so subject 4 is read like the others.
BCI2A_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# ---- Jia 2019 -----------------------------------------------------------------
JIA2019_SUBJECTS = tuple(range(1, 16))
JIA2019_TRIAL_SECONDS = 6.832        # 1,708 samples at 250 Hz; the shortest published epoch is 3,500 samples at 512 Hz
JIA2019_SAMPLE_SECONDS = JIA2019_TRIAL_SECONDS    # the whole epoch, as MOABB and PA-TCNet use it


@dataclass(frozen=True)
class DataConfig:
    # ---- which samples ------------------------------------------------------
    dataset: str = "liu2024"
    class_names: tuple[str, str] = ("rest", "imagery")      # label 0, label 1
    subjects: tuple[int, ...] = LIU2024_SUBJECTS
    num_channels: int = 29              # EEG electrodes, in the order of the dataset's reader
    sample_rate: int = 250
    trial_seconds: float = 8.0          # the stretch that is read and filtered
    sample_seconds: float = 1.9         # the stretch that is analysed

    # liu2024: two samples per trial, placed by the trial's own imagery onset (seconds from the
    # onset). The imagery lasts 4 s and the break 2 s; the first second of each is left out, and
    # sample_seconds is what remains of the break inside the epoch for a trial with the usual timing.
    imagery_start: float = 1.0
    rest_start: float = 4.0

    # bci2a and jia2019: one sample per trial; the analysed stretch starts cue_seconds after the
    # start of what is read (bci2a: the cue, 2 s after the trial start; jia2019: the epoch start).
    cue_seconds: float = 2.0
    session_indices: tuple[int, ...] | None = None    # None reads every session that contains trials

    # ---- signal -------------------------------------------------------------
    average_reference: bool = False           # subtract the mean over the electrodes (jia2019, as in PA-TCNet)
    band: tuple[float, float] = (8.0, 30.0)   # zero-phase Butterworth band-pass
    filter_order: int = 5

    # ---- node features: band power per electrode ----------------------------
    feature_band_edges: tuple[float, ...] = (8, 12, 16, 20, 24, 28, 32)
    welch_seconds: float = 2.0
    welch_overlap: float = 0.5

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}")
        if not self.subjects or len(set(self.subjects)) != len(self.subjects):
            raise ValueError("subjects must be a non-empty list without repeats")

    @property
    def trial_samples(self) -> int:
        return int(self.trial_seconds * self.sample_rate)

    @property
    def window_samples(self) -> int:
        """Samples of one analysed stretch."""
        return int(round(self.sample_seconds * self.sample_rate))

    @property
    def feature_bands(self) -> tuple[tuple[float, float], ...]:
        return tuple(zip(self.feature_band_edges[:-1], self.feature_band_edges[1:]))

    @property
    def task(self) -> str:
        return f"{self.class_names[1]} (1) against {self.class_names[0]} (0)"


def liu2024_config(**overrides) -> DataConfig:
    """Liu et al. 2024: 46 acute stroke patients, 29 electrodes; imagery against rest, two samples per trial."""
    return replace(DataConfig(), **overrides)


def bci2a_config(**overrides) -> DataConfig:
    """BCI Competition IV 2a: 9 healthy subjects, 22 electrodes; left against right hand, 144 trials each.

    A trial is read from 2 s before the cue to 2 s after it; the 2 s after the cue are analysed.
    """
    return replace(DataConfig(dataset="bci2a", class_names=("left hand", "right hand"), subjects=BCI2A_SUBJECTS,
                              num_channels=22, trial_seconds=4.0, sample_seconds=2.0), **overrides)


def jia2019_config(**overrides) -> DataConfig:
    """Jia 2019: 15 stroke patients, 63 electrodes without names; left against right hand, 80 trials each.

    Prepared as MOABB and PA-TCNet (Wang et al. 2026) do, so that results can be
    compared with theirs: the whole published epoch (0 to 6.8 s, resampled from
    512 Hz to 250 Hz, 1,708 samples) and a common average reference.
    """
    return replace(DataConfig(dataset="jia2019", class_names=("left hand", "right hand"), subjects=JIA2019_SUBJECTS,
                              num_channels=63, trial_seconds=JIA2019_TRIAL_SECONDS,
                              sample_seconds=JIA2019_SAMPLE_SECONDS, cue_seconds=0.0, average_reference=True),
                   **overrides)


def dataset_config(dataset: str, **overrides) -> DataConfig:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    return dict(liu2024=liu2024_config, bci2a=bci2a_config, jia2019=jia2019_config)[dataset](**overrides)
