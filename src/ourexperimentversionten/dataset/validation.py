"""Validate rules and data at pipeline boundaries."""
import math
import numpy as np
import torch
from scipy.signal import get_window


def validate_config(config):
    if not math.isfinite(config.sample_rate) or config.sample_rate <= 0:
        raise ValueError("sample_rate must be positive and finite")
    trials = config.trials
    if not trials.channel_indices or min(trials.channel_indices) < 0 or len(set(trials.channel_indices)) != len(trials.channel_indices):
        raise ValueError("channel_indices must be unique nonnegative indices")
    if not trials.session_indices or min(trials.session_indices) < 0:
        raise ValueError("session_indices must be nonnegative and nonempty")
    if not math.isfinite(trials.duration_seconds) or int(trials.duration_seconds * config.sample_rate) < 2:
        raise ValueError("Trial duration must provide at least two samples")
    mapping = dict(trials.label_map)
    if not mapping or len(mapping) != len(trials.label_map) or sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("label_map must map unique source labels to contiguous targets starting at zero")
    if trials.order not in ("legacy_interleave", "interleave", "grouped"):
        raise ValueError("Unknown trial order")
    if trials.incomplete_policy not in ("skip", "error"):
        raise ValueError("incomplete_policy must be skip or error")
    for edges in (config.preprocessing.edges, config.features.band_edges):
        if len(edges) < 2 or not all(math.isfinite(x) for x in edges) or not all(0 < a < b < config.sample_rate / 2 for a, b in zip(edges, edges[1:])):
            raise ValueError("Filter/band edges must increase strictly within (0, Nyquist)")
    if len(config.preprocessing.edges) != 2:
        raise ValueError("Preprocessing requires exactly two filter edges")
    for order in (config.preprocessing.order, config.features.filter_order):
        if not isinstance(order, int) or order < 1:
            raise ValueError("Filter orders must be positive integers")
    if not math.isfinite(config.features.welch_window_seconds) or config.features.welch_window_seconds * config.sample_rate < 2:
        raise ValueError("Welch window must contain at least two samples")
    if not 0 <= config.features.welch_overlap < 1:
        raise ValueError("Welch overlap must be in [0, 1)")
    if config.connectivity.method not in ("plv", "ciplv", "pli", "wpli", "coherence", "consensus", "rplv"):
        raise ValueError("Connectivity method must be plv, ciplv, pli, wpli, coherence, consensus, rplv")
    if not 0 <= config.connectivity.diagonal <= 1:
        raise ValueError("Connectivity diagonal must be in [0, 1]")
    if config.connectivity.method == "coherence":
        conn = config.connectivity
        validate_coherence_settings(config.sample_rate, conn.fmin, conn.fmax,
                                    conn.nperseg, conn.noverlap, conn.window, conn.detrend,
                                    int(trials.duration_seconds * config.sample_rate))
    graph = config.graphs
    if not math.isfinite(graph.threshold) or graph.mode not in ("legacy", "threshold") or graph.comparison not in ("gt", "ge"):
        raise ValueError("Invalid graph threshold, mode, or comparison")
    if graph.mode == "legacy" and (graph.comparison != "gt" or graph.self_loops or graph.directed):
        raise ValueError("Custom comparison/self-loop/direction rules require graph mode='threshold'")


def validate_session(session):
    if session.signal.ndim != 2 or len(session.positions) != len(session.labels):
        raise ValueError("Session requires a 2D signal and matching positions/labels")


def validate_trial(trial):
    if trial.signal.ndim != 2 or min(trial.signal.shape) == 0 or not np.isfinite(trial.signal).all():
        raise ValueError("Trial signal must be nonempty, finite, and [samples, channels]")
    if trial.label < 0:
        raise ValueError("Trial label must be nonnegative")


def validate_graph(graph):
    if graph.x.ndim != 2 or min(graph.x.shape) == 0 or not torch.isfinite(graph.x).all():
        raise ValueError("Graph features must be nonempty and finite")
    if graph.edge_index.dtype != torch.long or graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("edge_index must be an integer tensor of shape [2, edges]")
    if graph.edge_index.numel() and (graph.edge_index.min() < 0 or graph.edge_index.max() >= graph.x.shape[0]):
        raise ValueError("Graph edge references a missing node")
    if graph.y.dtype != torch.long or graph.y.numel() != 1 or graph.y.item() < 0:
        raise ValueError("Graph requires one nonnegative integer label")
    if graph.edge_weight is not None and (graph.edge_weight.shape != (graph.edge_index.shape[1],) or not torch.isfinite(graph.edge_weight).all()):
        raise ValueError("Edge weights must be finite and match edge count")


def validate_coherence_settings(sfreq, fmin, fmax, nperseg, noverlap, window, detrend, n_times):
    """Require a usable frequency band and at least two Welch segments."""
    if not math.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("Coherence sampling rate must be positive and finite")
    if not (math.isfinite(fmin) and math.isfinite(fmax) and 0 <= fmin < fmax <= sfreq / 2):
        raise ValueError("Coherence band must satisfy 0 <= fmin < fmax <= Nyquist")
    if isinstance(nperseg, bool) or not isinstance(nperseg, int) or nperseg < 2:
        raise ValueError("Coherence nperseg must be an integer of at least 2")
    overlap = nperseg // 2 if noverlap is None else noverlap
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < nperseg:
        raise ValueError("Coherence noverlap must be an integer in [0, nperseg)")
    if n_times < nperseg + (nperseg - overlap):
        raise ValueError("Coherence requires at least two Welch segments per trial")
    frequencies = np.fft.rfftfreq(nperseg, d=1 / sfreq)
    if not np.any((frequencies >= fmin) & (frequencies <= fmax)):
        raise ValueError("Coherence band contains no frequency bins")
    if detrend not in ("constant", "linear"):
        raise ValueError("Coherence detrend must be constant or linear")
    get_window(window, nperseg)  # Validate the SciPy window specification early.
