"""Create left- and right-hand imagery trials from continuous recordings."""

from typing import Any, Mapping

from braindecode.preprocessing import create_windows_from_events


DEFAULT_EVENT_MAPPING = {"left_hand": 0, "right_hand": 1}


def create_left_right_windows(
    dataset: Any,
    *,
    event_mapping: Mapping[str, int] = DEFAULT_EVENT_MAPPING,
    trial_start_offset_seconds: float = 0.0,
    trial_stop_offset_seconds: float = 0.0,
    preload: bool = True,
    n_jobs: int = 1,
) -> Any:
    """Create one window for each four-second left/right imagery annotation."""

    mapping = dict(event_mapping)
    if set(mapping) != set(DEFAULT_EVENT_MAPPING):
        raise ValueError("event_mapping must contain only left_hand and right_hand")
    sampling_rates = {recording.raw.info["sfreq"] for recording in dataset.datasets}
    if len(sampling_rates) != 1:
        raise ValueError("All recordings must have the same sampling frequency")
    sfreq = sampling_rates.pop()
    start_samples = round(trial_start_offset_seconds * sfreq)
    stop_samples = round(trial_stop_offset_seconds * sfreq)
    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=start_samples,
        trial_stop_offset_samples=stop_samples,
        mapping=mapping,
        preload=preload,
        n_jobs=n_jobs,
    )
