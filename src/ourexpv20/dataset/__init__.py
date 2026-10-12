"""Public interface for standalone EEG dataset preparation."""
from .config import DatasetConfig, default_dataset_config, legacy_dataset_config
from .alignment import align_trials
from .dataset import build_subject_dataset, build_trial_graph

__all__ = ["DatasetConfig", "default_dataset_config", "legacy_dataset_config", "build_subject_dataset", "build_trial_graph", "align_trials"]
