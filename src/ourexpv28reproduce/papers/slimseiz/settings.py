"""SlimSeiz's settings, from the paper (Lu et al. 2025) and its code (guoruilu/SlimSeiz).

Where the paper and the code disagree, the paper's value is used and the
code's is noted. Values that neither gives are assumptions; see README.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelSelectionSettings:
    """Section II-B and Loop_select_ch_PCA_SMOTE_DT.ipynb."""
    channels: int = 8                 # k: the paper's choice (Section III-B); the training notebook uses 6
    repetitions: int = 30             # m
    test_share: float = 0.3           # train_test_split(test_size=0.3) per repetition
    components: int = 60              # PCA(n_components=60), per channel
    downsample: int = 2               # the selection notebook takes every second sample
    smote_neighbours: int = 5         # imblearn's SMOTE default


@dataclass(frozen=True)
class TrainingSettings:
    """Loop_ADDNet_kFold.ipynb (the paper gives none of these)."""
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3       # Adam
    weight_decay: float = 1e-4
    contrastive_weight: float = 0.5   # loss = 0.5 cross-entropy + 0.5 supervised contrastive
    temperature: float = 0.08
    validation_share: float = 0.2     # 20 % of the training side, at random
    seed: int = 12345                 # the code sets none; ours
