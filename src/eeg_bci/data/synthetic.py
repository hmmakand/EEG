from __future__ import annotations

import mne
import numpy as np
from braindecode.datasets import create_from_X_y
from omegaconf import DictConfig
from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.types import DatasetInfo


def build_synthetic_dataset(
    dataset_cfg: DictConfig,
    preprocessing_cfg: DictConfig | None = None,
) -> tuple[BraindecodeLikeDataset, DatasetInfo]:
    mne.set_log_level("WARNING")
    rng = np.random.default_rng(7)
    n_trials = int(dataset_cfg.n_trials)
    n_channels = int(dataset_cfg.n_channels)
    n_times = int(dataset_cfg.n_times)
    n_classes = int(dataset_cfg.n_classes)
    sfreq = float(dataset_cfg.sfreq)

    y = np.arange(n_trials, dtype=np.int64) % n_classes
    x = rng.normal(
        loc=0.0,
        scale=float(dataset_cfg.noise_std),
        size=(n_trials, n_channels, n_times),
    ).astype(np.float32)

    time = np.arange(n_times, dtype=np.float32) / sfreq
    for class_id in range(n_classes):
        class_trials = y == class_id
        channel = class_id % n_channels
        frequency = 8.0 + (class_id * 4.0)
        x[class_trials, channel, :] += np.sin(2.0 * np.pi * frequency * time)

    ch_names = [f"EEG{idx:03d}" for idx in range(n_channels)]
    dataset = create_from_X_y(
        x,
        y,
        drop_last_window=False,
        sfreq=sfreq,
        ch_names=ch_names,
    )
    info = DatasetInfo(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        sfreq=sfreq,
    )
    return dataset, info
