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
    # Used only when the model has a sparse mask (ModelConfig.sparse_mask):
    mask_sparsity: float = 0.03        # weight of the mask penalty in the loss (beta of ASA-STGCN)
    mask_learning_rate: float = 0.01   # the mask has its own Adam learning rate and no weight decay
    device: str = "auto"          # "auto", "cpu" or "cuda"
    show_progress: bool = True

    def validate(self) -> None:
        for name in ("epochs", "batch_size", "ensemble"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mask_sparsity < 0 or not self.mask_learning_rate > 0:
            raise ValueError("mask_sparsity must be non-negative and mask_learning_rate positive")
        if not self.learning_rate > 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32 - self.ensemble:
            raise ValueError("seed must be a non-negative integer")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
