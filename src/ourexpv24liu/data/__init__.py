"""Recordings -> aligned trials -> graphs."""
from .config import (BCI2A_ALL_SUBJECTS, DATASETS, LIU2024_ALL_SUBJECTS, LIU2024_DUPLICATES, LIU2024_SUBJECTS, DataConfig, bci2a_config, dataset_config,
                     liu2024_config)
from .graph import build_graph
from .recordings import Trial, read_trials, subject_file
from .signal import align_subject


def build_subject_graphs(data_dir, subject: int, config: DataConfig):
    """All left/right trials of one subject as graphs, in recording order."""
    trials = align_subject(read_trials(data_dir, subject, config), config)
    return [build_graph(trial, config) for trial in trials]


__all__ = ["DataConfig", "DATASETS", "BCI2A_ALL_SUBJECTS", "LIU2024_SUBJECTS", "LIU2024_ALL_SUBJECTS",
           "LIU2024_DUPLICATES", "liu2024_config", "bci2a_config",
           "dataset_config", "Trial", "read_trials", "subject_file", "align_subject", "build_graph",
           "build_subject_graphs"]
