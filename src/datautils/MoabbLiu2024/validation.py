"""Validation checks for raw and processed Liu2024 recordings."""

from collections import Counter
from typing import Any


EXPECTED_EVENTS = {"left_hand", "right_hand"}


def validate_liu2024(
    dataset: Any,
    *,
    require_eeg_only: bool = False,
    expected_sfreq: float | None = None,
) -> list[dict[str, Any]]:
    """Validate recordings and return a compact per-recording summary."""

    if not getattr(dataset, "datasets", None):
        raise ValueError("The Liu2024 dataset has no recordings")

    summaries = []
    reference_channels: tuple[str, ...] | None = None
    for index, recording in enumerate(dataset.datasets):
        raw = recording.raw
        descriptions = set(map(str, raw.annotations.description))
        missing_events = EXPECTED_EVENTS - descriptions
        if missing_events:
            raise ValueError(f"Recording {index} is missing events: {sorted(missing_events)}")

        type_counts = Counter(raw.get_channel_types())
        if not type_counts.get("eeg"):
            raise ValueError(f"Recording {index} contains no EEG channels")
        if require_eeg_only and set(type_counts) != {"eeg"}:
            raise ValueError(f"Recording {index} is not EEG-only: {dict(type_counts)}")
        if expected_sfreq is not None and abs(raw.info["sfreq"] - expected_sfreq) > 1e-6:
            raise ValueError(
                f"Recording {index} has {raw.info['sfreq']} Hz; expected {expected_sfreq} Hz"
            )

        eeg_channels = tuple(
            name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"
        )
        if reference_channels is None:
            reference_channels = eeg_channels
        elif eeg_channels != reference_channels:
            raise ValueError(f"Recording {index} has inconsistent EEG channels or ordering")

        event_counts = Counter(map(str, raw.annotations.description))
        description = getattr(recording, "description", {})
        summaries.append(
            {
                "recording": index,
                "subject": description.get("subject"),
                "session": description.get("session"),
                "run": description.get("run"),
                "sfreq": float(raw.info["sfreq"]),
                "n_eeg_channels": len(eeg_channels),
                "left_hand_trials": event_counts["left_hand"],
                "right_hand_trials": event_counts["right_hand"],
            }
        )
    return summaries
