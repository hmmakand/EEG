"""Motor-area features of a trial, used to check pseudo-labels (train/calibration.py).

For every trial: the 8-30 Hz log power of the mean signal over the left, the
right and the midline motor electrodes, and the left-right difference. The
features are computed on the signal before alignment, where electrodes still
mean what their names say, and are standardised within the subject, which
needs no labels.
"""
import numpy as np

from .config import DataConfig
from .recordings import Trial
from .signal import analysed_signal

BCI2A_CHANNELS = ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                  "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz")
# The same electrodes in both datasets that have channel names.
ROI_GROUPS = dict(left=("FC3", "C3", "CP3"), right=("FC4", "C4", "CP4"), midline=("FCz", "Cz"))
ROI_FEATURES = 4      # left, right, midline, left minus right


def channel_names(dataset: str):
    """Electrode names in the order of the reader, or None when the dataset has none (jia2019)."""
    if dataset == "bci2a":
        return BCI2A_CHANNELS
    if dataset == "liu2024":
        from .liu2024 import EEG_CHANNELS
        return EEG_CHANNELS
    return None


def roi_features(trials: list[Trial], config: DataConfig):
    """[trials, 4], standardised over the trials of the subject. `trials` must not be aligned."""
    names = channel_names(config.dataset)
    if names is None:
        raise ValueError(f"{config.dataset} has no channel names; motor-area features cannot be computed")
    groups = [[names.index(name) for name in ROI_GROUPS[group]] for group in ("left", "right", "midline")]
    features = np.empty((len(trials), ROI_FEATURES))
    for row, trial in enumerate(trials):
        window = analysed_signal(trial, config)
        power = [np.log(window[:, group].mean(axis=1).var()) for group in groups]
        features[row] = (*power, power[0] - power[1])
    return (features - features.mean(axis=0)) / features.std(axis=0).clip(min=1e-12)
