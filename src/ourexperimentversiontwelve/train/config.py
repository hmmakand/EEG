"""Training settings; dataset rules remain in dataset/config.py."""
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    subjects: tuple[int, ...] = (1, 2, 3, 5, 6, 7, 8, 9)
    n_folds: int = 10
    split_shuffle: bool = True
    split_seed: int = 42
    fold_seed: int = 12345
    batch_size: int = 32
    train_shuffle: bool = False
    test_shuffle: bool = False
    learning_rate: float = 0.001
    # The original range(1, 250) executes exactly 249 epochs.
    num_epochs: int = 249
    hidden_channels: int = 22
    heads: int = 3
    training_engine: str = "supervised"
    fixmatch_lambda_u: float = 1.0
    fixmatch_confidence_threshold: float = 0.95
    fixmatch_weak_noise_std: float = 0.05
    fixmatch_strong_noise_std: float = 0.2
    fixmatch_strong_mask_prob: float = 0.3
    fixmatch_reuse_labeled_as_unlabeled: bool = False
    optimizer: str = "adam"
    loss: str = "cross_entropy"
    selection_policy: str = "best_test_accuracy"
    device: str = "auto"
    show_progress: bool = True
    output_path: Path | None = None  # None selects a filename from the connectivity method.


def default_training_config() -> TrainingConfig:
    return TrainingConfig()


def validate_training_config(config: TrainingConfig) -> None:
    for name in ("n_folds", "batch_size", "num_epochs", "hidden_channels", "heads"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if not config.subjects or any(isinstance(s, bool) or not isinstance(s, int) or s < 1 for s in config.subjects):
        raise ValueError("subjects must contain positive integers")
    if len(set(config.subjects)) != len(config.subjects):
        raise ValueError("subjects must not contain duplicates")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    for name in ("split_seed", "fold_seed"):
        seed = getattr(config, name)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError(f"{name} must be an integer in [0, 2**32)")
    if config.training_engine not in ("supervised", "fixmatch"):
        raise ValueError("training_engine must be supervised or fixmatch")
    if config.training_engine == "fixmatch":
        from .fixmatch_engine import validate_fixmatch_settings
        validate_fixmatch_settings(config.fixmatch_lambda_u, config.fixmatch_confidence_threshold,
                                   config.fixmatch_weak_noise_std, config.fixmatch_strong_noise_std,
                                   config.fixmatch_strong_mask_prob)
    elif config.fixmatch_reuse_labeled_as_unlabeled:
        raise ValueError("fixmatch_reuse_labeled_as_unlabeled requires training_engine='fixmatch'")
    if config.optimizer != "adam":
        raise ValueError("Only optimizer='adam' is implemented")
    if config.loss != "cross_entropy":
        raise ValueError("Only loss='cross_entropy' is implemented")
    if config.selection_policy != "best_test_accuracy":
        raise ValueError("Only selection_policy='best_test_accuracy' is implemented")
    if config.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
