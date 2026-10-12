"""Recordings -> aligned trials -> graphs."""
import torch

from .config import (BCI2A_ALL_SUBJECTS, DATASETS, JIA2019_SUBJECTS, jia2019_config, LIU2024_ALL_SUBJECTS, LIU2024_DUPLICATES, LIU2024_SUBJECTS, DataConfig, bci2a_config, dataset_config,
                     liu2024_config)
from .graph import build_graph
from .recordings import Trial, read_trials, subject_file
from .roi import channel_names, roi_features
from .signal import align_subject, low_frequency_signal


def build_subject_graphs(data_dir, subject: int, config: DataConfig):
    """All left/right trials of one subject as graphs, in recording order."""
    trials = read_trials(data_dir, subject, config)
    graphs = [build_graph(trial, config) for trial in align_subject(trials, config)]
    if config.low_band is not None:
        # The low band is aligned on its own: its covariance differs from that of the main band.
        for graph, trial in zip(graphs, align_subject(trials, config, low=True)):
            graph.signal_low = torch.tensor(low_frequency_signal(trial, config).T.copy(), dtype=torch.float32)
    if channel_names(config.dataset) is not None:
        # Motor-area features of every trial, [1, 4]; used only by the optional target calibration.
        for graph, features in zip(graphs, torch.tensor(roi_features(trials, config), dtype=torch.float32)):
            graph.roi = features.unsqueeze(0)
    return graphs


__all__ = ["DataConfig", "DATASETS", "BCI2A_ALL_SUBJECTS", "LIU2024_SUBJECTS", "LIU2024_ALL_SUBJECTS",
           "LIU2024_DUPLICATES", "liu2024_config", "bci2a_config", "jia2019_config", "JIA2019_SUBJECTS",
           "dataset_config", "Trial", "read_trials", "subject_file", "align_subject", "build_graph",
           "build_subject_graphs"]
