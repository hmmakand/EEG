"""EEG Conformer's settings, from the paper (Song et al. 2023) and its code (eeyhsong/EEG-conformer).

The paper gives the architecture, the optimiser and the augmentation; the
code gives what the paper does not (epochs, batch size, the classifier's
sizes). Values that neither gives are assumptions; see README.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    """Section II-C, Table I and conformer.py."""
    kernels: int = 40                 # k: temporal and spatial convolution kernels; also the token size
    temporal_kernel: int = 25         # (1, 25)
    pool: int = 75                    # average pooling (1, 75), stride (1, 15)
    pool_stride: int = 15
    depth: int = 6                    # N self-attention layers
    heads: int = 10
    dropout: float = 0.5              # convolution module, attention, feed-forward
    forward_expansion: int = 4
    hidden: tuple[int, int] = (256, 32)          # classifier, conformer.py
    hidden_dropout: tuple[float, float] = (0.5, 0.3)


@dataclass(frozen=True)
class PreparationSettings:
    """Section II-B and preprocessing/BCIIV2a.m: 6th-order Chebyshev type II band-pass 4-40 Hz (60 dB),
    zero-phase (filtfilt) on every trial; z-score with the training data's mean and standard deviation."""
    band: tuple[float, float] = (4.0, 40.0)
    order: int = 6
    stopband_db: float = 60.0


@dataclass(frozen=True)
class TrainingSettings:
    """Section III-C and conformer.py."""
    epochs: int = 2000                # conformer.py; the paper shows 2,000 epochs in Fig. 2
    batch_size: int = 72              # conformer.py
    learning_rate: float = 2e-4       # paper
    betas: tuple[float, float] = (0.5, 0.999)   # paper
    segments: int = 8                 # Ns of S&R (paper)
    seed: int = 12345                 # the code draws a random seed per subject; ours
