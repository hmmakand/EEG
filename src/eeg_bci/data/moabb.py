"""MOABB dataset integration for Braindecode experiments.

This module follows the standard Braindecode/MOABB data flow: set the local
MOABB/MNE download directory, load a named MOABB dataset with selected subjects,
apply configured Braindecode preprocessors, create event windows, and return the
windowed dataset plus shape metadata needed by the model factory.

Use a single-subject config for quick real-data smoke tests. Use the full
subject list with `scripts/braindecode_scripts/train_within_subjects.py` for within-subject training and evaluation runs,
where each subject is trained/evaluated separately.
"""

from __future__ import annotations

from typing import Any

import moabb
from braindecode.datasets import MOABBDataset
from omegaconf import DictConfig, OmegaConf
from eeg_bci.data.paths import resolve_data_dir
from eeg_bci.data.preprocessing import apply_preprocessing
from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.types import DatasetInfo
from eeg_bci.data.windowing import create_event_windows, infer_window_info


def build_moabb_dataset(
    dataset_cfg: DictConfig,
    preprocessing_cfg: DictConfig,
) -> tuple[BraindecodeLikeDataset, DatasetInfo]:
    """Load, preprocess, and window a configured MOABB dataset.

    `dataset.subject_ids` controls which subjects are fetched. Passing one
    subject is useful for fast pipeline checks; passing all dataset subjects is
    the expected setup for a full within-subject run.
    """

    data_dir = resolve_data_dir(dataset_cfg.get("data_dir"))
    moabb.set_download_dir(str(data_dir))

    subject_ids = _subject_ids_from_config(dataset_cfg.get("subject_ids"))
    dataset = MOABBDataset(
        dataset_name=str(dataset_cfg.dataset_name),
        subject_ids=subject_ids,
    )

    apply_preprocessing(dataset, preprocessing_cfg)
    sfreq = float(dataset.datasets[0].raw.info["sfreq"])
    windows = create_event_windows(dataset, dataset_cfg, preprocessing_cfg)
    n_chans, n_outputs, n_times = infer_window_info(windows)

    info = DatasetInfo(
        n_chans=n_chans,
        n_outputs=n_outputs,
        n_times=n_times,
        sfreq=sfreq,
    )
    return windows, info


def _subject_ids_from_config(value: Any) -> list[int] | int | None:
    """Normalize Hydra subject-id config for `MOABBDataset`."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return [int(subject_id) for subject_id in value]

    resolved = OmegaConf.to_container(value, resolve=True)
    if resolved is None:
        return None
    if isinstance(resolved, int):
        return resolved
    if isinstance(resolved, list):
        return [int(subject_id) for subject_id in resolved]

    raise TypeError(
        "dataset.subject_ids must be an int, a list of ints, or null; "
        f"got {type(resolved).__name__}."
    )
