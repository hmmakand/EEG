"""Training and evaluation settings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    hidden_channels: int = 22     # GATv2 width per head
    heads: int = 3
    epochs: int = 20
    batch_size: int = 32          # training batches are shuffled
    learning_rate: float = 0.001  # Adam
    weight_decay: float = 0.001
    seed: int = 12345             # the models of a fold use seed, seed + 1, ...
    ensemble: int = 3             # models per fold; their class probabilities are averaged
    center_logits: bool = True    # subtract the held-out subject's mean logit before deciding
    device: str = "auto"          # "auto", "cpu" or "cuda"
    show_progress: bool = True
    shuffle_training_labels: bool = False  # control run: training labels carry no information

    def validate(self) -> None:
        for name in ("hidden_channels", "heads", "epochs", "batch_size", "ensemble"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.learning_rate > 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32 - self.ensemble:
            raise ValueError("seed must be a non-negative integer")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
