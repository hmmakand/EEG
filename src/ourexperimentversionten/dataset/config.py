"""Dataset rules. Coherence connectivity by default with a graph threshold of 0.35."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrialConfig:
    session_indices: tuple[int, ...] = tuple(range(3, 9))
    channel_indices: tuple[int, ...] = tuple(range(22))
    label_map: tuple[tuple[int, int], ...] = ((1, 0), (2, 1))
    duration_seconds: float = 4.0
    event_index_offset: int = 0
    incomplete_policy: str = "skip"
    order: str = "legacy_interleave"


@dataclass(frozen=True)
class FilterConfig:
    edges: tuple[float, float] = (8, 30)
    order: int = 5


@dataclass(frozen=True)
class FeatureConfig:
    band_edges: tuple[float, ...] = tuple(range(8, 41, 4))
    filter_order: int = 5
    welch_window_seconds: float = 2.0
    welch_window: str = "hann"
    welch_overlap: float = 0.5
    detrend: str = "constant"
    inclusive_boundaries: bool = True


@dataclass(frozen=True)
class ConnectivityConfig:
    method: str = "rplv"
    diagonal: float = 0.0
    # Used only by coherence; sampling rate comes from DatasetConfig.sample_rate.
    fmin: float = 8.0
    fmax: float = 30.0
    nperseg: int = 256
    noverlap: int | None = None  # None means 50% overlap.
    window: str = "hann"
    detrend: str = "constant"


@dataclass(frozen=True)
class GraphConfig:
    threshold: float = 0.35
    # legacy preserves the two threshold passes and (0, 0) placeholders.
    mode: str = "legacy"
    comparison: str = "gt"
    self_loops: bool = False
    directed: bool = False
    include_edge_weights: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    sample_rate: float = 250
    trials: TrialConfig = field(default_factory=TrialConfig)
    preprocessing: FilterConfig = field(default_factory=FilterConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    graphs: GraphConfig = field(default_factory=GraphConfig)


def default_dataset_config() -> DatasetConfig:
    return DatasetConfig()
