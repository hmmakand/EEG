"""Dataset rules.

Field defaults are the legacy rules (whole 4 s trial, eight bands, PLV graph
thresholded at 0.35). `default_dataset_config()` returns the version-eighteen
rules: post-cue window, six bands, complete PLV-weighted graph.
"""
from dataclasses import dataclass, field, replace


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
    # Seconds of the filtered trial kept for connectivity and features;
    # None keeps the whole trial. Trials start 2 s before the cue.
    window_seconds: tuple[float, float] | None = None


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
    method: str = "plv"
    diagonal: float = 0.0


@dataclass(frozen=True)
class GraphConfig:
    threshold: float = 0.35
    # legacy preserves the two threshold passes and (0, 0) placeholders.
    # complete keeps every electrode pair and ignores the threshold.
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


def legacy_dataset_config() -> DatasetConfig:
    """Rules of the earlier versions, kept for the GATv2 baseline."""
    return DatasetConfig()


def default_dataset_config() -> DatasetConfig:
    """Version-eighteen rules.

    - Only seconds 2-4 of each trial (after the cue) are analysed.
    - Six bands from 8 to 32 Hz; the 8-30 Hz filter leaves nothing above.
    - Every electrode pair is an edge and carries its PLV as edge weight.
    """
    legacy = DatasetConfig()
    return replace(
        legacy,
        preprocessing=replace(legacy.preprocessing, window_seconds=(2.0, 4.0)),
        features=replace(legacy.features, band_edges=tuple(range(8, 33, 4))),
        graphs=replace(legacy.graphs, mode="complete", include_edge_weights=True))
