"""M-ShallowConvNet's settings, from the paper (Kim et al. 2022) and its code (comojin1994/m-shallowconvnet).

Where the paper and the code disagree, the paper's value is used and the
code's is noted. Values that neither gives are assumptions; they are listed
in this folder's README.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSettings:
    """Section II-D and model/mshallowconvnet.py, at 250 Hz."""
    kernels: int = 24                     # d, temporal and spatial convolutions
    temporal_kernel_seconds: float = 0.12 # fs x 0.12 = 30 samples
    pool_seconds: float = 0.3             # fs x 0.3 = 75 samples
    pool_overlap: float = 0.7             # stride int(75 x 0.3) = 22 samples
    conv_max_norm: float = 2.0
    dense_max_norm: float = 0.5           # applied once, as the code does (see network.py)
    dropout: float = 0.5
    log_floor: float = 1e-6


@dataclass(frozen=True)
class PreparationSettings:
    """Section II-B: low-pass 38 Hz with a 4th-order Butterworth filter, exponential moving standardisation
    with alpha 0.999; the code adds microvolts and standardises every analysed sample with an initial block
    of 4 s."""
    low_pass: float = 38.0
    filter_order: int = 4                 # paper; the code uses MNE's FIR filter
    standardize_factor: float = 1e-3      # 1 - alpha
    standardize_init_block: int = 1000    # 4 s at 250 Hz (code)
    standardize_eps: float = 1e-4         # braindecode default


@dataclass(frozen=True)
class TrainingSettings:
    """Section II-E and configs/bcicompet2a_config.yaml."""
    epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 2e-3           # Adam, cosine annealing to 0 over the epochs
    weight_decay: float = 0.075           # code (configs); not in the paper
    models: int = 5                       # five-fold cross-validation: five models, bagged on the test data
    seed: int = 42                        # code (configs)
    # CutCat: largest share of a sample replaced by a sample of the other class (lambda). The paper sets 0.125
    # for BCI IV 2a and 0.1 for its two-class datasets (2b and Zhou).
    cutcat_ratio: dict = None

    def __post_init__(self):
        if self.cutcat_ratio is None:
            object.__setattr__(self, "cutcat_ratio", dict(bci2a=0.125, liu2024=0.1, jia2019=0.1))
