"""UA-DAAN's settings, from the paper (Shen et al. 2025) and the reference implementation (chenjlbj).

The paper gives the structure, the losses and the training settings (Section
IV-C); layer sizes, the prior, the number of Monte Carlo samples, the loss
scales and the gradient-reversal schedule come from the reference
implementation. Values that neither gives are assumptions; see README.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSettings:
    """Section IV-B/D: band-pass 1-40 Hz, differential entropy in five bands. The band edges are not given."""
    bands: tuple[tuple[float, float], ...] = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 40))   # delta .. gamma
    filter_order: int = 4             # zero-phase Butterworth (assumption)


@dataclass(frozen=True)
class ModelSettings:
    """extractor.py, classifier.py, discriminator.py."""
    hidden: tuple[int, int, int] = (256, 64, 16)
    dropout: float = 0.2              # paper and code
    classifier_hidden: int = 4
    prior_mu: float = 0.0
    prior_sigma: float = 0.1
    discriminator_hidden: int = 4
    monte_carlo: int = 10             # model.py: 10 forward passes averaged


@dataclass(frozen=True)
class TrainingSettings:
    """Section IV-C and ua-daan12fold.py."""
    epochs: int = 30
    source_batch: int = 98
    target_batch: int = 8
    learning_rate: float = 1e-3       # Adam
    weight_decay: float = 1e-3
    classification_weight: float = 2.0    # model.py: ce * 2 + kl * 0.01
    kl_weight: float = 0.01
    global_scale: float = 0.1             # discriminator.py: 0.5 (source + target) * 0.1
    local_scale: float = 0.05             # ... * 0.05
    reversal_gamma: float = 1.0           # LambdaSheduler(gamma=1.0, max_iter=1000)
    reversal_iterations: int = 1000
    seed: int = 1234                      # ua-daan12fold.py: seed_torch(seed=1234)
