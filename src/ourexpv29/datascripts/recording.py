"""A sample, and the choice of reader for the configured dataset."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .config import DataConfig


@dataclass(frozen=True)
class Sample:
    signal: np.ndarray  # [samples, channels]: the whole stretch the sample is taken from
    start: int          # sample of `signal` at which the analysed stretch starts
    label: int          # 0 or 1; see DataConfig.class_names
    subject_id: int
    trial_id: int       # trial of the subject the sample comes from; samples of one trial stay together
    sample_id: int      # unique within the subject


def _reader(config: DataConfig):
    from . import bci2a, jia2019, liu2024      # the readers import Sample from this module
    return dict(liu2024=liu2024, bci2a=bci2a, jia2019=jia2019)[config.dataset]


def subject_file(data_dir, subject: int, config: DataConfig) -> Path:
    return _reader(config).subject_file(data_dir, subject)


def read_samples(data_dir, subject: int, config: DataConfig) -> list[Sample]:
    """All samples of one subject, in recording order."""
    samples = _reader(config).read_samples(data_dir, subject, config)
    if config.average_reference:
        samples = [replace(s, signal=s.signal - s.signal.mean(axis=1, keepdims=True)) for s in samples]
    if not samples:
        raise ValueError(f"Subject {subject}: no usable trial")
    for sample in samples:
        if (sample.signal.shape[1] != config.num_channels or sample.start < 0
                or sample.start + config.window_samples > len(sample.signal)):
            raise ValueError(f"Subject {subject}, sample {sample.sample_id}: the analysed stretch does not fit "
                             f"({config.num_channels} channels, {config.window_samples} samples from "
                             f"{sample.start} in {sample.signal.shape})")
    return samples
