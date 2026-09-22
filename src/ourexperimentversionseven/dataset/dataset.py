"""Data loading and graph construction for BCI Competition IV Dataset 2a.

Ported as-is from src/ourexperimentversionseven/kaggleeeg.ipynb (cells
565863bc, 8bad463b, 57c9aca1) -- the cells that actually produced
BCI_IV_2a_GAT_Results.json. No behavior changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from os.path import join as pjoin
from pathlib import Path
from typing import cast

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.signal as sig
import torch
from scipy.integrate import simpson

FS = 250  # sampling frequency (Hz)
BAND_FILTER = [8, 30]  # mu/beta bandpass for connectivity + raw signal
BANDS = list(range(8, 41, 4))  # [8, 12, 16, 20, 24, 28, 32, 36, 40]


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not locate the repository root above {start}")


def resolve_data_dir() -> str:
    kaggle_input = Path("/kaggle/input/bci-competition-iv-data-sets-2a")
    if kaggle_input.exists():
        return str(kaggle_input)
    return str(find_repo_root(Path.cwd()) / "data" / "kaggle" / "bci-competition-iv-data-sets-2a")


def load_bci_competition_data(data_dir, subject_number):
    """Load BCI Competition IV Dataset 2a and convert to the expected format."""
    mat_fname = pjoin(data_dir, f"A0{subject_number}T.mat")
    mat_contents = sio.loadmat(mat_fname)
    data = mat_contents["data"]

    left_trials = []
    right_trials = []

    for session_idx in range(3, 9):  # sessions 3-8 contain test data
        session_data = data[0, session_idx][0, 0]

        eeg_data = session_data[0]  # (time_samples, 25_channels)
        trial_positions = session_data[1].flatten()
        trial_labels = session_data[2].flatten()

        eeg_data = eeg_data[:, :22]  # EEG only, drop EOG channels

        trial_length = 1000  # 4 seconds * 250 Hz

        for start_pos, label in zip(trial_positions, trial_labels):
            start_sample = int(start_pos)
            end_sample = start_sample + trial_length

            if end_sample <= eeg_data.shape[0]:
                trial_data = eeg_data[start_sample:end_sample, :]

                if label == 1:  # left hand
                    left_trials.append(trial_data)
                elif label == 2:  # right hand
                    right_trials.append(trial_data)
                # feet (3) and tongue (4) are skipped

    subject_data = {
        "L": np.empty((1, len(left_trials)), dtype=object),
        "R": np.empty((1, len(right_trials)), dtype=object),
    }
    for i, trial in enumerate(left_trials):
        subject_data["L"][0, i] = trial
    for i, trial in enumerate(right_trials):
        subject_data["R"][0, i] = trial

    return subject_data


def bandpass(data: np.ndarray, edges: Sequence[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, "bandpass", fs=sample_rate, output="sos")
    return sig.sosfiltfilt(sos, data, axis=0)


def plvfcn(eegData):
    numElectrodes = eegData.shape[1]
    numTimeSteps = eegData.shape[0]
    plvMatrix = np.zeros((numElectrodes, numElectrodes))
    for electrode1 in range(numElectrodes):
        for electrode2 in range(electrode1 + 1, numElectrodes):
            phase1 = np.angle(cast(np.ndarray, sig.hilbert(eegData[:, electrode1])))
            phase2 = np.angle(cast(np.ndarray, sig.hilbert(eegData[:, electrode2])))
            phase_difference = phase2 - phase1
            plv = np.abs(np.sum(np.exp(1j * phase_difference)) / numTimeSteps)
            plvMatrix[electrode1, electrode2] = plv
            plvMatrix[electrode2, electrode1] = plv
    return plvMatrix


def compute_plv(subject_data):
    idx = ["L", "R"]
    numElectrodes = subject_data["L"][0, 0].shape[1]

    num_trials_L = subject_data["L"].shape[1]
    num_trials_R = subject_data["R"].shape[1]

    plv = {
        "L": np.zeros((numElectrodes, numElectrodes, num_trials_L)),
        "R": np.zeros((numElectrodes, numElectrodes, num_trials_R)),
    }

    for field in idx:
        num_trials = subject_data[field].shape[1]
        for j in range(num_trials):
            x = subject_data[field][0, j]
            plv[field][:, :, j] = plvfcn(x)

    l, r = plv["L"], plv["R"]
    yl = np.zeros((num_trials_L, 1))
    yr = np.ones((num_trials_R, 1))
    img = np.concatenate((l, r), axis=2)
    y = np.concatenate((yl, yr), axis=0)
    y = torch.tensor(y, dtype=torch.long)
    return img, y


def create_graphs(plv, threshold):
    graphs = []
    for i in range(plv.shape[2]):
        G = nx.Graph()
        G.add_nodes_from(range(plv.shape[0]))
        for u in range(plv.shape[0]):
            for v in range(plv.shape[0]):
                if u != v and plv[u, v, i] > threshold:
                    G.add_edge(u, v, weight=plv[u, v, i])
        graphs.append(G)
    return graphs


def aggregate_eeg_data(S1, band):
    idx = ["L", "R"]
    numElectrodes = S1["L"][0, 0].shape[1]

    max_sizes = {field: 0 for field in idx}
    for field in idx:
        for i in range(S1[field].shape[1]):
            max_sizes[field] = max(max_sizes[field], S1[field][0, i].shape[0])

    l = np.zeros((max_sizes["L"], numElectrodes, S1["L"].shape[1]))
    r = np.zeros((max_sizes["R"], numElectrodes, S1["R"].shape[1]))

    for i in range(S1["L"].shape[1]):
        x = S1["L"][0, i]
        l[: x.shape[0], :, i] = x

    for i in range(S1["R"].shape[1]):
        x = S1["R"][0, i]
        r[: x.shape[0], :, i] = x

    l = l[..., np.newaxis]
    l = np.tile(l, (1, 1, 1, len(band) - 1))

    r = r[..., np.newaxis]
    r = np.tile(r, (1, 1, 1, len(band) - 1))

    return l, r


def bandpass1(data: np.ndarray, edges: Sequence[float], sample_rate: float, poles: int = 5):
    sos = sig.butter(poles, edges, "bandpass", fs=sample_rate, output="sos")
    return sig.sosfiltfilt(sos, data)


def bandpower(data, low, high, fs=250):
    win = 2 * fs
    freqs, psd = sig.welch(data, fs, nperseg=min(win, len(data)))
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    return simpson(psd[idx_band], dx=freq_res)


def bandpowercalc(l, band, fs):
    x = np.zeros([l.shape[0], l.shape[3], l.shape[2]])
    for i in range(l.shape[0]):  # node/channel
        for j in range(l.shape[2]):  # trial/sample
            for k in range(l.shape[3]):  # frequency band
                data = l[i, :, j, k]
                low = band[k]
                high = band[k + 1]
                x[i, k, j] = bandpower(data, low, high, fs)
    return x
