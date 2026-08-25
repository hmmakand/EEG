"""Continuous-signal preprocessing for Liu2024."""

from typing import Any

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess,
)

from .config import (
    DEFAULT_H_FREQ,
    DEFAULT_L_FREQ,
    DEFAULT_SFREQ,
    DEFAULT_STANDARDIZE_INIT_SECONDS,
)


def preprocess_liu2024(
    dataset: Any,
    *,
    l_freq: float = DEFAULT_L_FREQ,
    h_freq: float = DEFAULT_H_FREQ,
    target_sfreq: float = DEFAULT_SFREQ,
    standardize: bool = True,
    factor_new: float = 1e-3,
    init_block_size: int = round(DEFAULT_STANDARDIZE_INIT_SECONDS * DEFAULT_SFREQ),
    n_jobs: int = 1,
) -> Any:
    """Band-pass, resample, and optionally standardize EEG recordings in place."""

    if not 0 <= l_freq < h_freq:
        raise ValueError("Expected 0 <= l_freq < h_freq")
    if target_sfreq <= 0:
        raise ValueError("target_sfreq must be positive")

    preprocessors = [
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, apply_on_array=False),
        Preprocessor("resample", sfreq=target_sfreq, apply_on_array=False),
    ]
    if standardize:
        preprocessors.append(
            Preprocessor(
                exponential_moving_standardize,
                factor_new=factor_new,
                init_block_size=init_block_size,
            )
        )
    preprocess(dataset, preprocessors, n_jobs=n_jobs)
    return dataset
