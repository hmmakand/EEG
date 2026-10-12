"""Leave-one-subject-out and within-subject training and evaluation."""
from .config import TrainConfig
from .loso import run_fold, run_loso
from .within import run_within, within_config

__all__ = ["TrainConfig", "run_fold", "run_loso", "run_within", "within_config"]
