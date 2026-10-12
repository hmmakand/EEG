"""Recordings -> aligned trials -> graphs."""
from .config import ALL_SUBJECTS, DataConfig
from .graph import build_graph
from .recordings import Trial, read_trials, subject_file
from .signal import align_subject


def build_subject_graphs(data_dir, subject: int, config: DataConfig | None = None):
    """All left/right trials of one subject as graphs, in recording order."""
    config = config or DataConfig()
    trials = read_trials(data_dir, subject, config)
    aligned = align_subject(trials, config)
    return [build_graph(trial, aligned_trial, config) for trial, aligned_trial in zip(trials, aligned)]


__all__ = ["DataConfig", "ALL_SUBJECTS", "Trial", "read_trials", "subject_file", "align_subject", "build_graph",
           "build_subject_graphs"]
