"""Recording -> samples -> aligned samples -> graphs."""
import torch

from .config import (BCI2A_SUBJECTS, DATASETS, JIA2019_SUBJECTS, LIU2024_ALL_SUBJECTS, LIU2024_DUPLICATES,
                     LIU2024_SUBJECTS, LIU2024_TOO_FEW_TRIALS, DataConfig, bci2a_config, dataset_config, jia2019_config,
                     liu2024_config)
from .graph import build_graph
from .recording import Sample, read_samples, subject_file
from .roi import CHANNEL_NAMES, roi_features
from .signal import align_subject


def build_subject_graphs(data_dir, subject: int, config: DataConfig | None = None):
    """All samples of one subject as graphs, in recording order."""
    config = config or liu2024_config()
    samples = read_samples(data_dir, subject, config)
    graphs = [build_graph(sample, config) for sample in align_subject(samples, config)]
    # Motor-area features of every sample, [1, 4]; used by the target calibration.
    # jia2019 has no channel names and therefore none.
    if config.dataset not in CHANNEL_NAMES:
        return graphs
    for graph, features in zip(graphs, torch.tensor(roi_features(samples, config), dtype=torch.float32)):
        graph.roi = features.unsqueeze(0)
    return graphs


__all__ = ["DataConfig", "DATASETS", "liu2024_config", "bci2a_config", "dataset_config", "LIU2024_SUBJECTS",
           "LIU2024_ALL_SUBJECTS", "LIU2024_DUPLICATES", "LIU2024_TOO_FEW_TRIALS", "BCI2A_SUBJECTS", "JIA2019_SUBJECTS", "jia2019_config", "Sample",
           "read_samples", "subject_file", "align_subject", "build_graph", "build_subject_graphs"]
