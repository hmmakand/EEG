"""Self-contained data profiles for the CFSPMNet comparison matrix."""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from eeg_bci.cfspmnet.canonicalization import LIU2024_FIGSHARE_EEG_CHANNELS
from eeg_bci.cfspmnet.figshare_liu2024 import (
    FIGSHARE_ARTICLE_ID,
    FIGSHARE_PARTICIPANTS_FILE_ID,
    FIGSHARE_PARTICIPANTS_MD5,
    FIGSHARE_PARTICIPANTS_SIZE,
    FIGSHARE_PARTICIPANTS_URL,
    FIGSHARE_RAW_FILE_ID,
    FIGSHARE_RAW_MD5,
    FIGSHARE_RAW_SIZE,
    FIGSHARE_RAW_URL,
    FIGSHARE_VERSION,
    ensure_figshare_participants,
    ensure_figshare_raw_archive,
    load_and_preprocess_figshare_subject,
)
from eeg_bci.cfspmnet.protocols import materialize_window_trials
from eeg_bci.data.adapters import BraindecodeLikeDataset
from eeg_bci.data.moabb import build_moabb_dataset
from eeg_bci.data.types import DatasetInfo


class _RawWithChannelNames(Protocol):
    ch_names: list[str]


class _WindowDatasetWithRaw(Protocol):
    raw: _RawWithChannelNames


class _WindowedBraindecodeDataset(BraindecodeLikeDataset, Protocol):
    datasets: Sequence[_WindowDatasetWithRaw]

    def get_metadata(self) -> pd.DataFrame: ...


PROJECT_MOABB = "project_moabb"
PAPER_ALIGNED_FIGSHARE = "paper_aligned_figshare"
PreprocessingName = Literal["project_moabb", "paper_aligned_figshare"]
PROJECT_PROFILE_VERSION = "project-moabb-v1"
PAPER_ALIGNED_PROFILE_VERSION = "paper-aligned-figshare-v2"
PROFILE_IMPLEMENTATION_VERSIONS = {
    PROJECT_MOABB: PROJECT_PROFILE_VERSION,
    PAPER_ALIGNED_FIGSHARE: PAPER_ALIGNED_PROFILE_VERSION,
}


@dataclass(frozen=True)
class PreprocessingProfile:
    name: PreprocessingName
    data_source: str
    sfreq: float
    n_chans: int
    n_times: int
    operations: tuple[str, ...]
    limitations: tuple[str, ...] = ()


PROJECT_PROFILE = PreprocessingProfile(
    name=PROJECT_MOABB,
    data_source="moabb_preprocessed_edf",
    sfreq=500.0,
    n_chans=29,
    n_times=2000,
    operations=(
        "MOABB Liu2024 processed EDF",
        "EEG-only channel selection",
        "volts_to_microvolts",
        "8-30 Hz project filter",
        "exponential_moving_standardization",
        "4 s MI event window",
    ),
    limitations=(
        "CPz acquisition reference is absent from the processed EDF",
        "EDF already has source-dataset mean removal and 0.5-40 Hz FIR filtering",
    ),
)

PAPER_ALIGNED_PROFILE = PreprocessingProfile(
    name=PAPER_ALIGNED_FIGSHARE,
    data_source="figshare_v5_raw_mat",
    sfreq=250.0,
    n_chans=30,
    n_times=1000,
    operations=(
        "2nd-order zero-phase Butterworth 8-30 Hz on each raw 8 s trial",
        "full-trial polyphase downsample from 500 Hz to 250 Hz",
        "30-electrode common-average reference",
        "last 1 s pre-MI baseline correction",
        "per-subject EOG-guided FastICA on the full 8 s trial when enabled",
        "4 s MI crop",
    ),
    limitations=(
        "CFSPMNet paper does not state its ICA component-selection parameters",
        "CFSPMNet paper does not state its baseline interval",
        "uses explicit reproducible choices documented in run metadata",
        "--skip-ica is diagnostic-only and produces a distinct fingerprint",
    ),
)


@dataclass(frozen=True)
class LoadedTrials:
    x: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    channel_names: tuple[str, ...]
    dataset_info: DatasetInfo
    preprocessing_metadata: dict[str, Any]


def load_trials_for_profile(
    *,
    repo_root: Path,
    subject_ids: list[int],
    profile_name: PreprocessingName,
    seed: int,
    apply_ica: bool = True,
    download_if_missing: bool = True,
    figshare_cache_dir: Path | None = None,
) -> LoadedTrials:
    """Load one complete cohort using the selected immutable data profile."""

    _validate_subject_ids(subject_ids)
    if profile_name == PROJECT_MOABB:
        return _load_project_trials(repo_root, subject_ids)
    if profile_name == PAPER_ALIGNED_FIGSHARE:
        cache_dir = repo_root / "data" / "figshare" / "liu2024"
        if figshare_cache_dir is not None:
            cache_dir = Path(figshare_cache_dir)
            if not cache_dir.is_absolute():
                cache_dir = repo_root / cache_dir
        return _load_paper_aligned_trials(
            subject_ids,
            cache_dir=cache_dir,
            seed=seed,
            apply_ica=apply_ica,
            download_if_missing=download_if_missing,
        )
    raise ValueError(
        f"Unsupported preprocessing profile {profile_name!r}; expected "
        f"{PROJECT_MOABB!r} or {PAPER_ALIGNED_FIGSHARE!r}."
    )


def _load_project_trials(repo_root: Path, subject_ids: list[int]) -> LoadedTrials:
    dataset_cfg = cast(
        DictConfig, OmegaConf.load(repo_root / "configs/dataset/liu2024.yaml")
    )
    preprocessing_cfg = cast(
        DictConfig, OmegaConf.load(repo_root / "configs/preprocessing/liu2024.yaml")
    )
    dataset_cfg.subject_ids = list(subject_ids)
    base_windows, dataset_info = build_moabb_dataset(dataset_cfg, preprocessing_cfg)
    windows = cast(_WindowedBraindecodeDataset, base_windows)
    metadata = windows.get_metadata().reset_index(drop=True)
    channel_names = tuple(windows.datasets[0].raw.ch_names)
    x, y, trial_subject_ids = materialize_window_trials(
        windows, metadata, channel_names
    )
    _assert_profile_shape(PROJECT_PROFILE, dataset_info, channel_names, x)
    return LoadedTrials(
        x=x,
        y=y,
        subject_ids=trial_subject_ids,
        channel_names=channel_names,
        dataset_info=dataset_info,
        preprocessing_metadata={
            "profile": asdict(PROJECT_PROFILE),
            "implementation_version": PROJECT_PROFILE_VERSION,
            "ica_enabled": False,
            "dataset_config": OmegaConf.to_container(
                dataset_cfg,
                resolve=True,
            ),
            "preprocessing_config": OmegaConf.to_container(
                preprocessing_cfg,
                resolve=True,
            ),
        },
    )


def _load_paper_aligned_trials(
    subject_ids: list[int],
    *,
    cache_dir: Path,
    seed: int,
    apply_ica: bool,
    download_if_missing: bool,
) -> LoadedTrials:
    participants_path = ensure_figshare_participants(
        cache_dir, download_if_missing=download_if_missing
    )
    archive_path = ensure_figshare_raw_archive(
        cache_dir, download_if_missing=download_if_missing
    )
    n_trials = len(subject_ids) * 40
    x = np.empty((n_trials, 30, 1000), dtype=np.float32)
    y = np.empty(n_trials, dtype=np.int64)
    trial_subject_ids = np.empty(n_trials, dtype=np.int64)
    audits: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive_path, "r") as archive:
        for offset, subject_id in enumerate(subject_ids):
            subject_x, subject_y, audit = load_and_preprocess_figshare_subject(
                archive,
                subject_id,
                apply_ica=apply_ica,
                seed=seed,
            )
            start = offset * 40
            stop = start + 40
            x[start:stop] = subject_x
            y[start:stop] = subject_y
            trial_subject_ids[start:stop] = subject_id
            audits.append(asdict(audit))

    dataset_info = DatasetInfo(
        n_chans=30,
        n_outputs=2,
        n_times=1000,
        sfreq=250.0,
    )
    _assert_profile_shape(
        PAPER_ALIGNED_PROFILE,
        dataset_info,
        LIU2024_FIGSHARE_EEG_CHANNELS,
        x,
    )
    return LoadedTrials(
        x=x,
        y=y,
        subject_ids=trial_subject_ids,
        channel_names=LIU2024_FIGSHARE_EEG_CHANNELS,
        dataset_info=dataset_info,
        preprocessing_metadata={
            "profile": asdict(PAPER_ALIGNED_PROFILE),
            "implementation_version": PAPER_ALIGNED_PROFILE_VERSION,
            "source": {
                "article_id": FIGSHARE_ARTICLE_ID,
                "version": FIGSHARE_VERSION,
                "raw_archive": {
                    "file_id": FIGSHARE_RAW_FILE_ID,
                    "url": FIGSHARE_RAW_URL,
                    "path": str(archive_path),
                    "size_bytes": FIGSHARE_RAW_SIZE,
                    "md5": FIGSHARE_RAW_MD5,
                },
                "participants": {
                    "file_id": FIGSHARE_PARTICIPANTS_FILE_ID,
                    "url": FIGSHARE_PARTICIPANTS_URL,
                    "path": str(participants_path),
                    "size_bytes": FIGSHARE_PARTICIPANTS_SIZE,
                    "md5": FIGSHARE_PARTICIPANTS_MD5,
                },
            },
            "parameters": {
                "input_units": "microvolts",
                "output_units": "microvolts",
                "operation_order": [
                    "bandpass",
                    "downsample",
                    "common_average_reference",
                    "baseline_correction",
                    "ica",
                    "mi_crop",
                ],
                "bandpass": {
                    "low_hz": 8.0,
                    "high_hz": 30.0,
                    "order": 2,
                    "design": "butterworth_sos_zero_phase",
                },
                "resampling": {
                    "method": "scipy.signal.resample_poly",
                    "input_hz": 500.0,
                    "output_hz": 250.0,
                    "scope": "complete_8_second_trial",
                },
                "reference": "common_average_over_30_eeg_channels",
                "baseline": {
                    "interval_seconds_relative_to_mi": [-1.0, 0.0],
                    "samples_at_250_hz": [250, 500],
                },
                "mi_crop": {
                    "interval_seconds": [0.0, 4.0],
                    "samples_at_250_hz": [500, 1500],
                },
                "ica": {
                    "enabled": apply_ica,
                    "method": "fastica",
                    "n_components": 0.99,
                    "eog_threshold": 3.0,
                    "seed_rule": "base_seed_plus_subject_id",
                },
            },
            "subject_audits": audits,
        },
    )


def _assert_profile_shape(
    profile: PreprocessingProfile,
    dataset_info: DatasetInfo,
    channel_names: tuple[str, ...],
    x: np.ndarray,
) -> None:
    actual = (
        dataset_info.sfreq,
        dataset_info.n_chans,
        dataset_info.n_times,
        len(channel_names),
        x.shape[1],
        x.shape[2],
    )
    expected = (
        profile.sfreq,
        profile.n_chans,
        profile.n_times,
        profile.n_chans,
        profile.n_chans,
        profile.n_times,
    )
    if actual != expected:
        raise ValueError(
            f"Preprocessing profile {profile.name!r} produced {actual}; "
            f"expected {expected}."
        )
    if not np.isfinite(x).all():
        raise ValueError(
            f"Preprocessing profile {profile.name!r} produced non-finite trials."
        )


def _validate_subject_ids(subject_ids: list[int]) -> None:
    if len(subject_ids) < 2:
        raise ValueError("A LOSO cohort requires at least two subjects.")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Cohort subject IDs must be unique.")
    invalid = [subject_id for subject_id in subject_ids if subject_id not in range(1, 51)]
    if invalid:
        raise ValueError(f"Liu2024 subject IDs must be in 1..50; got {invalid}.")
