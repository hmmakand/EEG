"""MSCFormer's settings, from the paper (Zhao et al. 2025) and its code (snailpt/MSCFormer).

Where the paper and the code disagree, the paper's value is used and the
code's is noted. Values that neither gives are assumptions; see README.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    """Table 2 of the paper and MSCFormer_model.py."""
    temporal_filters: int = 16                    # F1 per branch; F2 = 3 x F1 = 48 is the token size
    kernels: tuple[int, int, int] = (85, 65, 45)  # temporal kernels of the three branches
    pool: int = 44                                # P: 44 for BCI IV 2a, 52 for BCI IV 2b (2 or 3 electrodes)
    heads: int = 8
    depth: int = 5
    dropout: float = 0.5                          # transformer
    branch_dropout_within: float = 0.5            # after each branch's pooling, subject-specific (paper, code)
    branch_dropout_cross: float = 0.25            # code, cross-subject ("0.25 for cross-subject")
    position_dropout: float = 0.1
    classifier_dropout: float = 0.25
    max_tokens: int = 100                         # length of the learned position embedding (code)


@dataclass(frozen=True)
class TrainingSettings:
    """Section "Experimental setup" of the paper and MSCFormer.ipynb."""
    epochs: int = 1000
    batch_size: int = 72          # real samples per step; with three times as many S&R samples, 288 (paper)
    augmentation_factor: int = 3  # S&R samples per real sample (code: number_aug; README: "three times")
    segments: int = 8             # Ns
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)
    weight_decay: float = 0.0     # 0 for BCI IV 2a (code); 0.001 only for BCI IV 2b's three electrodes (paper)
    models: int = 5               # five-fold cross-validation on the training side; results averaged
    seed: int = 12345             # the code draws a random seed per subject; ours
