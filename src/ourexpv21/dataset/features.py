"""Band filtering, spectral estimation, and node feature construction."""
import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson
from .preprocessing import apply_bandpass


def make_band_ranges(edges):
    return tuple(zip(edges[:-1], edges[1:]))


def compute_psd(signal, sample_rate, config):
    length = min(int(config.welch_window_seconds * sample_rate), len(signal))
    return welch(signal, fs=sample_rate, window=config.welch_window,
                 nperseg=length, noverlap=int(length * config.welch_overlap),
                 detrend=config.detrend)


def integrate_bandpower(freqs, psd, low, high, inclusive=True):
    mask = (freqs >= low) & ((freqs <= high) if inclusive else (freqs < high))
    if len(freqs) < 2 or not mask.any():
        raise ValueError("Insufficient frequency bins for bandpower")
    return simpson(psd[mask], dx=freqs[1] - freqs[0])


def compute_node_features(signal, sample_rate, config):
    """Input [samples, channels]; output [channels, bands * segments],
    ordered band by band with the segments of one band adjacent."""
    bands = make_band_ranges(config.band_edges)
    features = np.zeros((signal.shape[1], len(bands), config.segments))
    for band_index, (low, high) in enumerate(bands):
        filtered = apply_bandpass(signal.T, (low, high), sample_rate,
                                  config.filter_order, time_axis=-1)
        for segment_index, segment in enumerate(np.array_split(filtered, config.segments, axis=-1)):
            for channel in range(signal.shape[1]):
                freqs, psd = compute_psd(segment[channel], sample_rate, config)
                features[channel, band_index, segment_index] = integrate_bandpower(
                    freqs, psd, low, high, config.inclusive_boundaries)
    return features.reshape(signal.shape[1], -1)
