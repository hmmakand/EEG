"""Classifier settings only; subjects and fold assignments come from the bundle."""
from dataclasses import dataclass, replace
import math
from .config import default_training_config, validate_training_config, TrainingConfig

_STANDARD = default_training_config()


@dataclass(frozen=True)
class SynTrainingConfig:
    subjects: tuple[int, ...] | None = None  # None uses all saved subjects.
    num_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = _STANDARD.learning_rate
    weight_decay: float = _STANDARD.weight_decay
    model: str = _STANDARD.model
    model_paths: tuple[str, ...] = _STANDARD.model_paths
    hidden_channels: int = _STANDARD.hidden_channels
    heads: int = _STANDARD.heads
    model_seed: int = _STANDARD.fold_seed
    train_shuffle: bool = _STANDARD.train_shuffle
    # LOSO trainer only. `ensemble` models are trained with seeds model_seed,
    # model_seed + 1, ... and their probabilities averaged. `center_logits`
    # removes the held-out subject's mean logit before the decision (uses the
    # test subject's unlabeled trials).
    ensemble: int = 1
    center_logits: bool = False
    # LOSO trainer only: False drops the synthetic trials of the saved fold
    # and trains on the training subjects' original trials.
    use_synthetic: bool = True
    # LOSO trainer only: extra training loss against subject-specific features
    # (train/invariance.py): "none", "adversarial", "contrastive" or "proxy".
    invariance: str = "none"
    invariance_weight: float = 0.1
    device: str = _STANDARD.device
    show_progress: bool = _STANDARD.show_progress

    def standard(self) -> TrainingConfig:
        """Adapt to shared learning helpers; no splitting helper consumes this."""
        return replace(_STANDARD, subjects=self.subjects if self.subjects is not None else _STANDARD.subjects,
                       num_epochs=self.num_epochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
                       weight_decay=self.weight_decay,
                       model=self.model, model_paths=self.model_paths, hidden_channels=self.hidden_channels, heads=self.heads, fold_seed=self.model_seed,
                       train_shuffle=self.train_shuffle, device=self.device, show_progress=self.show_progress)

    def validate(self) -> None:
        validate_training_config(self.standard())
        if isinstance(self.ensemble, bool) or not isinstance(self.ensemble, int) or self.ensemble < 1:
            raise ValueError("ensemble must be a positive integer")
        if self.invariance not in ("none", "adversarial", "contrastive", "proxy"):
            raise ValueError("invariance must be none, adversarial, contrastive or proxy")
        if not math.isfinite(self.invariance_weight) or self.invariance_weight < 0:
            raise ValueError("invariance_weight must be non-negative and finite")
