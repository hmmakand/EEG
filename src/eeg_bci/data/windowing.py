"""Window creation helpers for event-based EEG decoding.

Braindecode trains on fixed-size windows rather than raw continuous recordings.
This module converts preprocessed event-based datasets into Braindecode windows,
normalizes Hydra config values into the exact types expected by Braindecode, and
infers model input dimensions from the resulting windows.
"""

from __future__ import annotations

from typing import Any

from braindecode.preprocessing import create_windows_from_events
from omegaconf import DictConfig, OmegaConf


def create_event_windows(dataset, dataset_cfg: DictConfig, preprocessing_cfg: DictConfig):
    """Create Braindecode windows from annotated EEG events.

    Window size, stride, trial offsets, preload behavior, and event-to-class
    mapping are controlled by Hydra config. A `None` window size keeps one
    window per trial, matching the common trial-wise BCI IV 2a setup.
    """

    sfreq = float(dataset.datasets[0].raw.info["sfreq"])
    window_size_samples = seconds_to_samples(preprocessing_cfg.window_size_s, sfreq)
    window_stride_samples = seconds_to_samples(preprocessing_cfg.window_stride_s, sfreq)
    trial_start_offset_samples = int(
        round(float(preprocessing_cfg.trial_start_offset_s) * sfreq)
    )
    trial_stop_offset_samples = int(
        round(float(preprocessing_cfg.trial_stop_offset_s) * sfreq)
    )
    mapping = mapping_from_config(dataset_cfg.get("mapping"))

    return create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False,
        mapping=mapping,
        preload=bool(preprocessing_cfg.preload),
    )


def infer_window_info(windows, *, n_outputs: int | None = None) -> tuple[int, int, int]:
    """Infer channel count, class count, and time samples from windows.

    ``n_outputs`` should be supplied from ``dataset.mapping`` (see
    :func:`n_outputs_from_mapping`) so the model head and metric label space
    stay stable even when a particular subject/fold is missing a class. When it
    is ``None`` the class count falls back to the observed unique targets.
    """

    sample_x, _, *_ = windows[0]
    if n_outputs is None:
        n_outputs = len(set(int(windows[idx][1]) for idx in range(len(windows))))
    return int(sample_x.shape[0]), int(n_outputs), int(sample_x.shape[-1])


def n_outputs_from_mapping(mapping: dict[str, int] | None) -> int | None:
    """Return the classifier head size implied by a label mapping.

    Returns ``max(target) + 1`` so it matches the ``range(n_outputs)`` label
    space used everywhere downstream (this assumes the 0-based contiguous
    targets that :func:`mapping_from_config` already produces). Returns ``None``
    when no mapping is configured, leaving the count to be inferred from data.
    """

    if not mapping:
        return None
    return int(max(mapping.values())) + 1


def seconds_to_samples(value: float | None, sfreq: float) -> int | None:
    """Convert a duration in seconds to samples, preserving `None`."""

    if value is None:
        return None
    return int(round(float(value) * sfreq))


def mapping_from_config(value: Any) -> dict[str, int] | None:
    """Normalize Hydra label mapping config for Braindecode windowing."""

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(label): int(target) for label, target in value.items()}

    resolved = OmegaConf.to_container(value, resolve=True)
    if resolved is None:
        return None
    if isinstance(resolved, dict):
        return {str(label): int(target) for label, target in resolved.items()}

    raise TypeError(
        "dataset.mapping must be a mapping from event label to integer target, "
        f"or null; got {type(resolved).__name__}."
    )
