"""Training and evaluation settings; those of version 23."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32          # training batches are shuffled
    learning_rate: float = 0.001  # Adam
    weight_decay: float = 0.001
    seed: int = 12345             # the models of a fold use seed, seed + 1, ...
    ensemble: int = 3             # models per fold; their class probabilities are averaged
    center_logits: bool = True    # subtract the held-out subject's mean logit before deciding
    # Calibration on the unlabelled test subject, leave one subject out only (train/calibration.py).
    # On by default: it is the configuration with the best measured result (see README). It is
    # switched off automatically for a dataset without channel names and for within-subject runs.
    target_calibration: bool = True
    # Epochs on the training subjects alone before it starts; None means half of the epochs.
    calibration_warmup_epochs: int | None = None
    calibration_source_weight: float = 0.95    # weight of the training subjects' loss; the rest goes to the test trials
    calibration_confidence: float = 0.90       # smallest probability for a prediction to become a target
    calibration_similarity_floor: float = 0.0  # lower bound of the motor-area similarity thresholds
    device: str = "auto"          # "auto", "cpu" or "cuda"
    show_progress: bool = True
    shuffle_training_labels: bool = False  # control run: the training labels carry no information

    @property
    def warmup_epochs(self) -> int:
        return self.epochs // 2 if self.calibration_warmup_epochs is None else self.calibration_warmup_epochs

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "ensemble"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_calibration and not (0 <= self.warmup_epochs <= self.epochs
                                            and 0 < self.calibration_source_weight <= 1
                                            and 0.5 <= self.calibration_confidence <= 1):
            raise ValueError("target calibration needs warm-up epochs of at most epochs, a source weight in (0, 1] "
                             "and a confidence in [0.5, 1]")
        if not self.learning_rate > 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32 - self.ensemble:
            raise ValueError("seed must be a non-negative integer")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
