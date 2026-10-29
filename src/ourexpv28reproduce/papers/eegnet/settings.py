"""EEGNet's settings, from the paper (Lawhern et al. 2018) and its code (arl-eegmodels).

Values that neither gives are assumptions; they are listed in this folder's
README.md, with the reason for each.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    """EEGNet-8,2: Table 2 of the paper and EEGNet() in original/EEGModels.py."""
    temporal_filters: int = 8        # F1
    depth: int = 2                   # D: spatial filters per temporal filter
    pointwise_filters: int = 16      # F2 = F1 * D
    kernel: int = 32                 # 64 (half the sampling rate) in Table 2; 32 for the SMR data, "since the data
                                     # were high-passed at 4 Hz" (section 2.3). Ours are high-passed at 4 Hz too.
    separable_kernel: int = 16       # 500 ms at 32 Hz
    depthwise_max_norm: float = 1.0
    dense_max_norm: float = 0.25     # norm_rate
    dropout_cross_subject: float = 0.25
    dropout_within_subject: float = 0.5


@dataclass(frozen=True)
class PreparationSettings:
    """Section 2.1.4 (SMR): resampled to 128 Hz, pre-processed as Schirrmeister et al. 2017 [32] with the
    authors' software (braindecode), band 4-40 Hz (Table 1)."""
    sample_rate: int = 128
    band: tuple[float, float] = (4.0, 40.0)
    filter_order: int = 3               # braindecode: causal Butterworth (assumption: the order of [32]'s code)
    standardize_factor: float = 1e-3    # braindecode exponential moving standardisation
    standardize_init_block: int = 1000  # samples at 250 Hz
    standardize_eps: float = 1e-4


@dataclass(frozen=True)
class TrainingSettings:
    """Section 2.2.1 and 2.3, and the code's example (examples/ERP.py) for what the paper does not give."""
    epochs: int = 500                   # "500 training iterations (epochs)"
    batch_size: int = 16                # not in the paper; examples/ERP.py
    learning_rate: float = 1e-3         # Adam with Keras defaults
    adam_epsilon: float = 1e-7          # Keras default (PyTorch: 1e-8)
    # Cross-subject validation: for SMR, 3 of the 8 other subjects (section 2.3); here the same share of the
    # other subjects, chosen at random per held-out subject.
    validation_fraction: float = 3 / 8
    seed: int = 12345                   # the paper gives none; our seed
