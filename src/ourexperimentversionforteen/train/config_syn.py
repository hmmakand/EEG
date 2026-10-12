"""Classifier settings only; subjects and fold assignments come from the bundle."""
from dataclasses import dataclass, replace
from .config import default_training_config, validate_training_config, TrainingConfig

_STANDARD = default_training_config()


@dataclass(frozen=True)
class SynTrainingConfig:
    subjects: tuple[int, ...] | None = None  # None uses all saved subjects.
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = _STANDARD.learning_rate
    hidden_channels: int = _STANDARD.hidden_channels
    heads: int = _STANDARD.heads
    model_seed: int = _STANDARD.fold_seed
    train_shuffle: bool = _STANDARD.train_shuffle
    device: str = _STANDARD.device
    show_progress: bool = _STANDARD.show_progress

    def standard(self) -> TrainingConfig:
        """Adapt to shared learning helpers; no splitting helper consumes this."""
        return replace(_STANDARD, subjects=self.subjects if self.subjects is not None else _STANDARD.subjects,
                       num_epochs=self.num_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
                       hidden_channels=self.hidden_channels, heads=self.heads, fold_seed=self.model_seed,
                       train_shuffle=self.train_shuffle, device=self.device, show_progress=self.show_progress)

    def validate(self) -> None:
        validate_training_config(self.standard())
