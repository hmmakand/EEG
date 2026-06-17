"""Preprocessing helpers for Braindecode EEG datasets.

This module translates Hydra preprocessing config into Braindecode
`Preprocessor` objects and applies them to loaded datasets. The default pipeline
keeps only EEG channels, band-pass filters motor-imagery frequencies, optionally
resamples, and optionally applies exponential moving standardization.
"""

from __future__ import annotations

import mne
from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)
from omegaconf import DictConfig


def build_preprocessors(preprocessing_cfg: DictConfig) -> list[Preprocessor]:
    """Build the ordered Braindecode preprocessing pipeline from config."""

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor(
            "filter",
            l_freq=preprocessing_cfg.low_cut_hz,
            h_freq=preprocessing_cfg.high_cut_hz,
        ),
    ]
    if preprocessing_cfg.resample_sfreq is not None:
        preprocessors.append(
            Preprocessor("resample", sfreq=float(preprocessing_cfg.resample_sfreq))
        )
    if preprocessing_cfg.standardize:
        preprocessors.append(
            Preprocessor(
                exponential_moving_standardize,
                factor_new=0.001,
                init_block_size=1000,
            )
        )
    return preprocessors


def apply_preprocessing(dataset, preprocessing_cfg: DictConfig) -> None:
    """Apply configured preprocessors in-place to a Braindecode dataset."""

    mne.set_log_level("WARNING")
    preprocess(dataset, build_preprocessors(preprocessing_cfg), n_jobs=1)
