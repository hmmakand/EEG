"""Synthetic training protocols; learning defaults come from the standard trainer."""
from dataclasses import dataclass, replace
from .config import default_training_config, validate_training_config, TrainingConfig

_STANDARD = default_training_config()


@dataclass(frozen=True)
class SynTrainingConfig:
    subjects: tuple[int, ...] = _STANDARD.subjects
    num_epochs: int = _STANDARD.num_epochs
    batch_size: int = _STANDARD.batch_size
    learning_rate: float = _STANDARD.learning_rate
    hidden_channels: int = _STANDARD.hidden_channels
    heads: int = _STANDARD.heads
    model_seed: int = _STANDARD.fold_seed
    mix_seed: int = _STANDARD.split_seed
    train_shuffle: bool = _STANDARD.train_shuffle
    device: str = _STANDARD.device
    show_progress: bool = _STANDARD.show_progress
    combine_valid_test: bool = False
    protocol: str = "fixed"
    n_folds: int = _STANDARD.n_folds

    def standard(self) -> TrainingConfig:
        """Supply the unchanged model/optimizer/loss helpers with their config type."""
        return replace(default_training_config(), subjects=self.subjects, num_epochs=self.num_epochs,
                       n_folds=self.n_folds, batch_size=self.batch_size, learning_rate=self.learning_rate,
                       hidden_channels=self.hidden_channels, heads=self.heads, fold_seed=self.model_seed,
                       split_seed=self.mix_seed, train_shuffle=self.train_shuffle,
                       device=self.device, show_progress=self.show_progress)

    def validate(self) -> None:
        validate_training_config(self.standard())
        if self.protocol not in ("fixed", "kfold"):
            raise ValueError("protocol must be fixed or kfold")
        if self.protocol == "kfold" and self.combine_valid_test:
            raise ValueError("combine-valid-test applies only to the fixed protocol")
