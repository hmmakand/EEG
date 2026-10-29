"""PA-TCNet's settings, from the paper (Wang et al. 2026, arXiv:2604.16554) and its code.

Where the paper and the code disagree, the paper's value is used, and the
code's value is noted. Values that neither gives are assumptions; they are
listed in this folder's README.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSettings:
    name: str                  # the paper's name for the dataset
    dataset: str               # our reader in datascripts/
    subjects: tuple[int, ...]
    channels: int              # model input [channels, samples] (paper, Table 1)
    samples: int
    window_seconds: tuple[float, float] | None   # analysed stretch, from the reader's onset; None = the whole epoch
    # Table 3 of the paper (the code's defaults in brackets)
    source_weight: float       # alpha: weight of the source loss after warm-up (code default 0.95)
    confidence: float          # tau_p: confidence threshold of a pseudo-label (code default 0.90)
    roi_floor: float           # delta_min: lower bound of the ROI similarity thresholds (code default 0.70)
    warmup_epochs: int         # source-only epochs before PGTC starts (code default 10)


# The 24 patients of the paper's XW-Stroke cohort: right-handed, first-ever stroke, NIHSS < 10,
# duration < 10 years (paper, section 3.1; also DEFAULT_XW_SUB_LIST in the code). 47 and 49 are the
# same recording (datascripts/liu2024.py); 5 is a copy of 4, which is not in the list.
XW_STROKE_SUBJECTS = (2, 5, 8, 9, 11, 12, 14, 17, 21, 23, 24, 26, 27, 28, 30, 32, 33, 37, 38, 43, 44, 47, 49, 50)

DATASETS = dict(
    # Extension: version 27 subjects/window; PGTC defaults from the released code.
    bci2a=DatasetSettings(name="BCI IV 2a (extension)", dataset="bci2a",
                          subjects=(1, 2, 3, 4, 5, 6, 7, 8, 9), channels=22, samples=500,
                          window_seconds=(0.0, 2.0), source_weight=0.95, confidence=0.90,
                          roi_floor=0.70, warmup_epochs=10),
    liu2024=DatasetSettings(name="XW-Stroke", dataset="liu2024", subjects=XW_STROKE_SUBJECTS, channels=30,
                            samples=1000, window_seconds=(0.0, 4.0),
                            source_weight=0.98, confidence=0.60, roi_floor=0.50, warmup_epochs=25),
    jia2019=DatasetSettings(name="2019-Stroke", dataset="jia2019", subjects=tuple(range(1, 16)), channels=63,
                            samples=1708, window_seconds=None,
                            source_weight=0.95, confidence=0.60, roi_floor=0.45, warmup_epochs=10),
)


@dataclass(frozen=True)
class ModelSettings:
    """Table 2 of the paper, and train.py for what the table does not give."""
    embedding: int = 30                       # Table 2
    depth: int = 2                            # Table 2
    temporal_filters_per_branch: int = 10     # Table 2
    spatial_multiplier: int = 3               # Table 2
    temporal_kernels: tuple[int, int, int] = (36, 24, 18)   # Table 2 (the model's default)
    pool: int = 8                             # Table 2; both pooling sizes
    dropout: float = 0.5                      # train.py default; not in the paper


@dataclass(frozen=True)
class TrainingSettings:
    """Section 3.4 of the paper, and train.py for what the paper does not give."""
    epochs: int = 200                 # maximum (paper)
    learning_rate: float = 0.001      # paper
    weight_decay: float = 0.001       # paper
    batch_size: int = 64              # train.py default
    patience: int = 30                # train.py: early stopping on the loss of the held-out subject
    seed: int = 2                     # train.py default


def flattened_features(samples: int, model: ModelSettings = ModelSettings()) -> int:
    """The input size of the classification head, as train.py computes it."""
    return (samples // model.pool // model.pool) * model.embedding


def dataset_settings(dataset: str, cohort: str = "version27") -> DatasetSettings:
    """Common comparison inputs by default; original paper inputs remain explicit."""
    from dataclasses import replace
    from ...datascripts.version27 import SUBJECTS
    settings = DATASETS[dataset]
    if cohort == "version27":
        dimensions = dict(liu2024=(29, 475), bci2a=(22, 500), jia2019=(63, 1708))
        channels, samples = dimensions[dataset]
        return replace(settings, subjects=SUBJECTS[dataset], channels=channels, samples=samples,
                       window_seconds=None)
    if cohort == "distinct" and dataset == "liu2024":
        return replace(settings, subjects=tuple(s for s in settings.subjects if s != 49))
    if cohort not in ("published", "distinct"):
        raise ValueError(f"Unknown cohort: {cohort}")
    return settings
