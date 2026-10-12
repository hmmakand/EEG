"""Every rule that turns a recording into graphs. Changing a value changes the experiment."""
from dataclasses import dataclass

ALL_SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9)   # adds subject 4, which the reported result leaves out


@dataclass(frozen=True)
class DataConfig:
    # ---- which trials -------------------------------------------------------
    subjects: tuple[int, ...] = (1, 2, 3, 5, 6, 7, 8, 9)    # the reported result; see ALL_SUBJECTS
    # Sessions to read. None reads every session that contains trials: the six
    # motor-imagery runs, at positions 3-8 for most subjects and 1-6 for subject 4.
    session_indices: tuple[int, ...] | None = None
    channel_indices: tuple[int, ...] = tuple(range(22))     # the 22 EEG electrodes
    label_map: tuple[tuple[int, int], ...] = ((1, 0), (2, 1))  # left hand -> 0, right hand -> 1
    sample_rate: int = 250
    trial_seconds: float = 4.0          # read from the trial start (2 s before the cue)

    # ---- signal -------------------------------------------------------------
    band: tuple[float, float] = (8.0, 30.0)   # zero-phase Butterworth band-pass
    filter_order: int = 5
    window_seconds: tuple[float, float] = (2.0, 4.0)   # analysed part, after the cue

    # ---- node features: log band power per electrode ------------------------
    feature_band_edges: tuple[float, ...] = (8, 12, 16, 20, 24, 28, 32)
    welch_seconds: float = 2.0
    welch_overlap: float = 0.5

    # ---- graph --------------------------------------------------------------
    plv_threshold: float = 0.35   # two electrodes are connected when their PLV is above this

    @property
    def trial_samples(self) -> int:
        return int(self.trial_seconds * self.sample_rate)

    @property
    def window_samples(self) -> tuple[int, int]:
        start, stop = (int(round(second * self.sample_rate)) for second in self.window_seconds)
        return start, stop

    @property
    def feature_bands(self) -> tuple[tuple[float, float], ...]:
        return tuple(zip(self.feature_band_edges[:-1], self.feature_band_edges[1:]))
