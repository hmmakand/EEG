"""Training and evaluation settings. The defaults are those of leave one subject out."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20              # fixed in advance; no validation set, no early stopping
    batch_size: int = 32          # training batches are shuffled
    learning_rate: float = 0.001  # Adam
    weight_decay: float = 0.001
    seed: int = 12345             # the models of a fold use seed, seed + 1, ...
    ensemble: int = 3             # models per fold; their class probabilities are averaged
    center_logits: bool = True    # subtract the held-out subject's mean logit before deciding
    # Calibration on the unlabelled test subject (train/calibration.py); leave one subject out only.
    target_calibration: bool = True
    calibration_warmup_epochs: int = 10        # epochs on the training subjects alone before it starts
    calibration_source_weight: float = 0.95    # weight of the training subjects' loss; the rest goes to the test samples
    calibration_confidence: float = 0.90       # smallest probability for a prediction to become a target
    calibration_similarity_floor: float = 0.0  # lower bound of the motor-area similarity thresholds
    device: str = "auto"          # "auto", "cpu" or "cuda"
    show_progress: bool = True

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "ensemble"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_calibration and not (0 <= self.calibration_warmup_epochs <= self.epochs and 0 < self.calibration_source_weight <= 1
                                            and 0.5 <= self.calibration_confidence <= 1):
            raise ValueError("calibration needs warm-up epochs of at most epochs, a source weight in (0, 1] "
                             "and a confidence in [0.5, 1]")
        if not self.learning_rate > 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32 - self.ensemble:
            raise ValueError("seed must be a non-negative integer")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
