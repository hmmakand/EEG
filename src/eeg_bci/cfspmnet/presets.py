"""Locked presets for the four CFSPMNet comparison experiments."""

from __future__ import annotations

from eeg_bci.cfspmnet.experiment import CFSPMExperimentConfig
from eeg_bci.cfspmnet.preprocessing import (
    PAPER_ALIGNED_FIGSHARE,
    PROJECT_MOABB,
)
from eeg_bci.cfspmnet.protocols import (
    AFFECTED_UNAFFECTED_FLIP,
    RAW_LEFT_RIGHT,
    SOURCE_ONLY,
    SPPM_TRANSDUCTIVE,
    CanonicalizationMode,
    TrainingProtocol,
)


def project_source_only_config() -> CFSPMExperimentConfig:
    return _project_config(
        experiment_name="cfspmnet_project_source_only_loso",
        model_name="cfspmnet_project_source_only",
        protocol=SOURCE_ONLY,
        canonicalization=RAW_LEFT_RIGHT,
    )


def canonical_source_only_config() -> CFSPMExperimentConfig:
    return _project_config(
        experiment_name="cfspmnet_canonical_source_only_loso",
        model_name="cfspmnet_canonical_source_only",
        protocol=SOURCE_ONLY,
        canonicalization=AFFECTED_UNAFFECTED_FLIP,
    )


def sppm_transductive_config() -> CFSPMExperimentConfig:
    return _project_config(
        experiment_name="cfspmnet_sppm_transductive_loso",
        model_name="cfspmnet_sppm_transductive",
        protocol=SPPM_TRANSDUCTIVE,
        canonicalization=AFFECTED_UNAFFECTED_FLIP,
    )


def paper_aligned_sppm_config() -> CFSPMExperimentConfig:
    return CFSPMExperimentConfig(
        experiment_name="cfspmnet_paper_aligned_sppm_loso",
        model_name="cfspmnet_paper_aligned_sppm",
        protocol=SPPM_TRANSDUCTIVE,
        canonicalization=RAW_LEFT_RIGHT,
        preprocessing=PAPER_ALIGNED_FIGSHARE,
        subject_ids=list(range(1, 51)),
        stage_i_epochs=25,
        stage_ii_epochs=175,
        alpha=0.98,
        pseudo_threshold=0.60,
        matching_tolerance_floor=0.50,
        batch_size=40,
        learning_rate=1e-3,
        weight_decay=1e-3,
        optimizer="adam",
        scheduler="none",
        seed=2,
        apply_ica=True,
    )


def _project_config(
    *,
    experiment_name: str,
    model_name: str,
    protocol: TrainingProtocol,
    canonicalization: CanonicalizationMode,
) -> CFSPMExperimentConfig:
    # All three project-preprocessed controls intentionally share this complete
    # recipe. Source-only continues through the Stage-II epoch budget, so the
    # canonical source-only versus SPPM contrast has equal training duration.
    return CFSPMExperimentConfig(
        experiment_name=experiment_name,
        model_name=model_name,
        protocol=protocol,
        canonicalization=canonicalization,
        preprocessing=PROJECT_MOABB,
        subject_ids=list(range(1, 51)),
        stage_i_epochs=25,
        stage_ii_epochs=175,
        alpha=0.98,
        pseudo_threshold=0.60,
        matching_tolerance_floor=0.50,
        batch_size=16,
        learning_rate=0.000625,
        weight_decay=0.0,
        optimizer="adamw",
        scheduler="cosine",
        seed=2,
        apply_ica=False,
    )
