"""Registry of selectable Graph-Liu2024-VersionTwo feature combinations.

Every combination module (``without_csd_alpha_wpli``, ``csd_alpha_wpli``, ...)
exposes the same shape -- ``load_dataset``, ``validate_dataset``,
``create_loso_dataloaders``, ``create_group_dataloaders``,
``create_within_subject_dataloaders``, ``fit_feature_normalization``, and
``GraphDataLoaderConfig`` -- so training code can select one by name instead
of importing a specific module. To add a new combination: write a sibling
module with that same shape, then add one entry below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch_geometric.loader import DataLoader

from src.datautils.graphdataversiontwo.saved_dataset import SavedDataset

from . import csd_alpha_wpli, csd_gamma_wpli, without_csd_alpha_wpli, without_csd_gamma_wpli
from .loso_split import FeatureNormalization, GraphDataLoaderConfig, LosoDataLoaderBundle


@dataclass(frozen=True)
class Combination:
    """One selectable node-variant/edge-variant/band pairing.

    ``load_dataset``, ``validate_dataset``, ``create_loso_dataloaders``,
    ``create_group_dataloaders``, ``create_within_subject_dataloaders``, and
    ``fit_feature_normalization`` are the corresponding module's functions
    of the same name -- see :mod:`without_csd_alpha_wpli` for their
    contract.
    """

    name: str
    node_variant: str
    edge_variant: str
    band_name: str
    load_dataset: Callable[[], SavedDataset]
    validate_dataset: Callable[[SavedDataset], None]
    create_loso_dataloaders: Callable[..., LosoDataLoaderBundle]
    create_group_dataloaders: Callable[..., tuple[DataLoader, DataLoader, FeatureNormalization]]
    create_within_subject_dataloaders: Callable[
        ..., tuple[DataLoader, DataLoader, FeatureNormalization]
    ]
    fit_feature_normalization: Callable[..., FeatureNormalization]


COMBINATIONS: dict[str, Combination] = {
    "without_csd_alpha_wpli": Combination(
        name="without_csd_alpha_wpli",
        node_variant=without_csd_alpha_wpli.NODE_VARIANT,
        edge_variant=without_csd_alpha_wpli.EDGE_VARIANT,
        band_name=without_csd_alpha_wpli.BAND_NAME,
        load_dataset=without_csd_alpha_wpli.load_dataset,
        validate_dataset=without_csd_alpha_wpli.validate_dataset,
        create_loso_dataloaders=without_csd_alpha_wpli.create_loso_dataloaders,
        create_group_dataloaders=without_csd_alpha_wpli.create_group_dataloaders,
        create_within_subject_dataloaders=without_csd_alpha_wpli.create_within_subject_dataloaders,
        fit_feature_normalization=without_csd_alpha_wpli.fit_feature_normalization,
    ),
    "csd_alpha_wpli": Combination(
        name="csd_alpha_wpli",
        node_variant=csd_alpha_wpli.NODE_VARIANT,
        edge_variant=csd_alpha_wpli.EDGE_VARIANT,
        band_name=csd_alpha_wpli.BAND_NAME,
        load_dataset=csd_alpha_wpli.load_dataset,
        validate_dataset=csd_alpha_wpli.validate_dataset,
        create_loso_dataloaders=csd_alpha_wpli.create_loso_dataloaders,
        create_group_dataloaders=csd_alpha_wpli.create_group_dataloaders,
        create_within_subject_dataloaders=csd_alpha_wpli.create_within_subject_dataloaders,
        fit_feature_normalization=csd_alpha_wpli.fit_feature_normalization,
    ),
    "without_csd_gamma_wpli": Combination(
        name="without_csd_gamma_wpli",
        node_variant=without_csd_gamma_wpli.NODE_VARIANT,
        edge_variant=without_csd_gamma_wpli.EDGE_VARIANT,
        band_name=without_csd_gamma_wpli.BAND_NAME,
        load_dataset=without_csd_gamma_wpli.load_dataset,
        validate_dataset=without_csd_gamma_wpli.validate_dataset,
        create_loso_dataloaders=without_csd_gamma_wpli.create_loso_dataloaders,
        create_group_dataloaders=without_csd_gamma_wpli.create_group_dataloaders,
        create_within_subject_dataloaders=without_csd_gamma_wpli.create_within_subject_dataloaders,
        fit_feature_normalization=without_csd_gamma_wpli.fit_feature_normalization,
    ),
    "csd_gamma_wpli": Combination(
        name="csd_gamma_wpli",
        node_variant=csd_gamma_wpli.NODE_VARIANT,
        edge_variant=csd_gamma_wpli.EDGE_VARIANT,
        band_name=csd_gamma_wpli.BAND_NAME,
        load_dataset=csd_gamma_wpli.load_dataset,
        validate_dataset=csd_gamma_wpli.validate_dataset,
        create_loso_dataloaders=csd_gamma_wpli.create_loso_dataloaders,
        create_group_dataloaders=csd_gamma_wpli.create_group_dataloaders,
        create_within_subject_dataloaders=csd_gamma_wpli.create_within_subject_dataloaders,
        fit_feature_normalization=csd_gamma_wpli.fit_feature_normalization,
    ),
}

DEFAULT_COMBINATION: str = "without_csd_alpha_wpli"


def get_combination(name: str) -> Combination:
    """Look up one registered combination by name, or raise ``ValueError``."""

    try:
        return COMBINATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown combination {name!r}; choose one of {sorted(COMBINATIONS)}"
        ) from None
