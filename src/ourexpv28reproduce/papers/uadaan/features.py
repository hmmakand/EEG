"""UA-DAAN's input: differential entropy (DE) per channel and band (Section IV-D).

DE of a band-passed signal under a Gaussian assumption is 0.5 * log(2 pi e var).
Each trial's whole stretch is band-passed once per band (zero-phase), the
sample is cut from it, and its variance gives the DE. The samples are version
27's (datascripts/version27.py); the feature vector of a sample has
channels x 5 values (channel-major). The reference implementation's input
was 640 values: 128 channels x 5 bands.
"""
import numpy as np
from scipy import signal

from ...datascripts import SAMPLE_RATE
from ...datascripts.version27 import read_samples
from .settings import FeatureSettings

MICROVOLT_READERS = ("bci2a",)


def differential_entropy(samples, dataset: str, settings: FeatureSettings = FeatureSettings()):
    """(features [samples, channels x bands] float32, labels, groups). Volts; the BCI IV 2a reader returns
    microvolts, which are converted, so that every dataset has the same units (DE depends on the scale)."""
    to_volts = 1e-6 if dataset in MICROVOLT_READERS else 1.0
    filters = [signal.butter(settings.filter_order, band, "bandpass", fs=SAMPLE_RATE, output="sos")
               for band in settings.bands]
    filtered, features = {}, []
    for sample in samples:
        key = id(sample.trial)
        if key not in filtered:
            stretch = sample.trial.signal * to_volts
            filtered[key] = [signal.sosfiltfilt(sos, stretch, axis=0) for sos in filters]
        window = slice(sample.start, sample.start + sample.length)
        variance = np.stack([band[window].var(axis=0) for band in filtered[key]], axis=1)    # [channels, bands]
        features.append(0.5 * np.log(2 * np.pi * np.e * variance).reshape(-1))
    return (np.stack(features).astype(np.float32), np.array([s.label for s in samples], dtype=np.int64),
            np.array([s.group for s in samples]))


def load_subject(data_dir, dataset: str, subject: int, settings: FeatureSettings = FeatureSettings()):
    return differential_entropy(read_samples(data_dir, dataset, subject), dataset, settings)
