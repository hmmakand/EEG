"""DBConformer's settings, from the paper (Wang et al. 2026) and its code (wzwvv/DBConformer).

Where the paper and the code disagree, the paper's value is used and the
code's is noted. Values that neither gives are assumptions; see README.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    """Section III and models/DBConformer.py with the scripts' flags (posemb, channel attention, both branches)."""
    embedding: int = 40               # D, also the temporal filters F
    temporal_kernel: int = 63         # depthwise temporal convolution of the temporal patch embedding
    patch: int = 125                  # W: 0.5 s at 250 Hz (the code's value for 250 Hz datasets)
    spatial_filters: int = 16         # spa_dim: Conv1d(1, 16, 25, stride 5) per channel
    spatial_kernel: int = 25
    spatial_stride: int = 5
    heads: int = 10                   # TransformerEncoderBlock default in the code
    dropout: float = 0.5              # patch embedding and transformer
    forward_expansion: int = 4
    hidden: tuple[int, int] = (64, 32)              # classifier
    hidden_dropout: tuple[float, float] = (0.5, 0.3)


# Transformer depth (temporal and spatial branch). DBConformer_LOSO.py: 2 for BNCI2014001 (our bci2a) and
# BNCI2014004, 6 for its other datasets (14 to 59 channels); DBConformer_CV.py: 2 for every dataset.
DEPTH = dict(loso=dict(bci2a=2, liu2024=6, jia2019=6), within=dict(bci2a=2, liu2024=2, jia2019=2))


@dataclass(frozen=True)
class PreparationSettings:
    """Section IV-A4: "the standard preprocessing steps in MOABB", then Euclidean alignment. MOABB's motor
    imagery paradigm band-passes 8-32 Hz with MNE's default IIR filter (4th-order Butterworth, forward and
    backward)."""
    band: tuple[float, float] = (8.0, 32.0)
    filter_order: int = 4
    rank_tolerance: float = 1e-10     # eigenvalues below this share of the largest are left out of the alignment


@dataclass(frozen=True)
class TrainingSettings:
    """Section IV-C3 and the scripts."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3       # Adam, PyTorch defaults otherwise
    seed: int = 1                     # the paper averages seeds 1 to 5; one run with the first
