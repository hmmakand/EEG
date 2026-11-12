"""Motor-area features of a sample, used to check pseudo-labels (train/calibration.py).

For every sample: the 8-30 Hz log power of the mean signal over the left, the
right and the midline motor electrodes, and the left-right difference. The
features are computed on the signal before alignment, where electrodes still
mean what their names say, and are standardised within the subject, which
needs no labels.
"""
import numpy as np

from .config import DataConfig
from . import bci2a, liu2024

# jia2019 has no channel names, so it has no motor-area features.
CHANNEL_NAMES = dict(liu2024=liu2024.EEG_CHANNELS, bci2a=bci2a.EEG_CHANNELS)
from .recording import Sample
from .signal import analysed_signal

# The same electrodes in both datasets.
ROI_GROUPS = dict(left=("FC3", "C3", "CP3"), right=("FC4", "C4", "CP4"), midline=("FCz", "Cz"))
ROI_FEATURES = 4      # left, right, midline, left minus right


def roi_features(samples: list[Sample], config: DataConfig):
    """[samples, 4], standardised over the samples of the subject. `samples` must not be aligned."""
    if config.dataset not in CHANNEL_NAMES:
        raise ValueError(f"{config.dataset} has no channel names; motor-area features cannot be computed")
    names = CHANNEL_NAMES[config.dataset]
    groups = [[names.index(name) for name in ROI_GROUPS[group]] for group in ("left", "right", "midline")]
    features = np.empty((len(samples), ROI_FEATURES))
    for row, sample in enumerate(samples):
        window = analysed_signal(sample, config)
        power = [np.log(window[:, group].mean(axis=1).var()) for group in groups]
        features[row] = (*power, power[0] - power[1])
    return (features - features.mean(axis=0)) / features.std(axis=0).clip(min=1e-12)
