"""Graph-Liu2024-VersionOne data contracts and LOSO loading for experiment four.

Multiple node/edge/band combinations are available as sibling modules (see
``combinations.py`` for the registry, e.g. ``without_csd_alpha_wpli`` and
``csd_alpha_wpli``). The flat names re-exported below are the *default*
combination (``without_csd_alpha_wpli``) for convenience and backward
compatibility; training code that must support switching combinations should
use ``get_combination``/``COMBINATIONS`` instead.
"""

from .combinations import COMBINATIONS, Combination, DEFAULT_COMBINATION, get_combination
from .loso_split import (
    FeatureNormalization,
    GraphDataLoaderConfig,
    LosoDataLoaderBundle,
    LosoGraphSplit,
    create_loso_splits,
)
from .validation import (
    BAND_NAME,
    DATASET_DIR,
    EDGE_VARIANT,
    EXPECTED_EDGES,
    EXPECTED_GRAPHS,
    EXPECTED_NODES,
    EXPECTED_NODE_FEATURES,
    EXPECTED_SUBJECT_IDS,
    NODE_FEATURE_NAMES,
    NODE_VARIANT,
    alpha_band_index,
    load_dataset,
    validate_dataset,
)
from .within_subject_split import WithinSubjectFold, create_within_subject_folds
from .without_csd_alpha_wpli import (
    create_loso_dataloaders,
    fit_feature_normalization,
)

__all__ = [
    "BAND_NAME",
    "COMBINATIONS",
    "Combination",
    "DATASET_DIR",
    "DEFAULT_COMBINATION",
    "EDGE_VARIANT",
    "EXPECTED_EDGES",
    "EXPECTED_GRAPHS",
    "EXPECTED_NODES",
    "EXPECTED_NODE_FEATURES",
    "EXPECTED_SUBJECT_IDS",
    "FeatureNormalization",
    "GraphDataLoaderConfig",
    "LosoDataLoaderBundle",
    "LosoGraphSplit",
    "NODE_FEATURE_NAMES",
    "NODE_VARIANT",
    "WithinSubjectFold",
    "alpha_band_index",
    "create_loso_dataloaders",
    "create_loso_splits",
    "create_within_subject_folds",
    "fit_feature_normalization",
    "get_combination",
    "load_dataset",
    "validate_dataset",
]
