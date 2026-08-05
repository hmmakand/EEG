"""Direct loader for Liu2024's raw Figshare MAT archive.

MOABB intentionally consumes Figshare's already processed EDF archive. The
paper-aligned experiment instead uses sourcedata.zip so it can retain the full
eight-second trial, CPz reference, and EOG channels before preprocessing.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import logging
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pooch
from mne.preprocessing import ICA
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt

from eeg_bci.cfspmnet.canonicalization import LIU2024_FIGSHARE_EEG_CHANNELS

logger = logging.getLogger(__name__)

FIGSHARE_ARTICLE_ID = 21679035
FIGSHARE_VERSION = 5
FIGSHARE_RAW_FILE_ID = 38516555
FIGSHARE_RAW_URL = (
    f"https://ndownloader.figshare.com/files/{FIGSHARE_RAW_FILE_ID}"
)
FIGSHARE_RAW_FILENAME = "sourcedata.zip"
FIGSHARE_RAW_SIZE = 1_874_824_439
FIGSHARE_RAW_MD5 = "4336b1ddbde42f89edb82f513a5c0721"
FIGSHARE_PARTICIPANTS_FILE_ID = 42345150
FIGSHARE_PARTICIPANTS_URL = (
    f"https://ndownloader.figshare.com/files/{FIGSHARE_PARTICIPANTS_FILE_ID}"
)
FIGSHARE_PARTICIPANTS_FILENAME = "participants.tsv"
FIGSHARE_PARTICIPANTS_SIZE = 3_557
FIGSHARE_PARTICIPANTS_MD5 = "d82596d5309f7d207c760918e74d4553"

RAW_SHAPE = (40, 33, 4000)
RAW_SFREQ = 500.0
OUTPUT_SFREQ = 250.0
MI_START_SAMPLE = 500
MI_STOP_SAMPLE = 1500
BASELINE_START_SAMPLE = 250
BASELINE_STOP_SAMPLE = 500
EOG_NAMES = ("HEOL", "VEOR")
RAW_DATA_CHANNEL_NAMES = (
    *LIU2024_FIGSHARE_EEG_CHANNELS,
    *EOG_NAMES,
    "MARKER",
)


@dataclass(frozen=True)
class FigshareSubjectAudit:
    subject_id: int
    archive_member: str
    ica_enabled: bool
    ica_excluded_components: tuple[int, ...]
    sample_rate_hz: float
    marker_codes: tuple[int, ...]
    baseline_samples: tuple[int, int]
    mi_samples: tuple[int, int]


def ensure_figshare_raw_archive(
    cache_dir: Path,
    *,
    download_if_missing: bool = True,
) -> Path:
    """Return the checksummed 1.87 GB raw archive, downloading only if allowed."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / FIGSHARE_RAW_FILENAME

    if archive_path.exists():
        _validate_archive_identity(archive_path)
        return archive_path
    if not download_if_missing:
        raise FileNotFoundError(
            f"Raw Liu2024 archive is missing at {archive_path}. Download "
            f"{FIGSHARE_RAW_URL} (MD5 {FIGSHARE_RAW_MD5}) or allow downloading."
        )

    logger.info(
        "Downloading Liu2024 raw Figshare archive (%.2f GiB) to %s",
        FIGSHARE_RAW_SIZE / 1024**3,
        archive_path,
    )
    retrieved = pooch.retrieve(
        url=FIGSHARE_RAW_URL,
        known_hash=f"md5:{FIGSHARE_RAW_MD5}",
        path=cache_dir,
        fname=FIGSHARE_RAW_FILENAME,
        progressbar=True,
    )
    archive_path = Path(retrieved)
    _validate_archive_identity(archive_path)
    return archive_path


def ensure_figshare_participants(
    cache_dir: Path,
    *,
    download_if_missing: bool = True,
) -> Path:
    """Return the checksummed Figshare participant metadata."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    participants_path = cache_dir / FIGSHARE_PARTICIPANTS_FILENAME

    if participants_path.exists():
        _validate_participants_identity(participants_path)
        return participants_path
    if not download_if_missing:
        raise FileNotFoundError(
            f"Liu2024 participant metadata is missing at {participants_path}. "
            f"Download {FIGSHARE_PARTICIPANTS_URL} or allow downloading."
        )

    retrieved = pooch.retrieve(
        url=FIGSHARE_PARTICIPANTS_URL,
        known_hash=f"md5:{FIGSHARE_PARTICIPANTS_MD5}",
        path=cache_dir,
        fname=FIGSHARE_PARTICIPANTS_FILENAME,
        progressbar=False,
    )
    participants_path = Path(retrieved)
    _validate_participants_identity(participants_path)
    return participants_path


def load_and_preprocess_figshare_subject(
    archive: zipfile.ZipFile,
    subject_id: int,
    *,
    apply_ica: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, FigshareSubjectAudit]:
    """Load one raw MAT member and return paper-aligned 4 s MI trials."""

    member = (
        f"sourcedata/sub-{subject_id:02d}/"
        f"sub-{subject_id:02d}_task-motor-imagery_eeg.mat"
    )
    try:
        payload = archive.read(member)
    except KeyError as error:
        raise FileNotFoundError(
            f"Subject {subject_id} is missing from {archive.filename}: {member}"
        ) from error

    rawdata, labels = _load_mat_payload(payload)
    _validate_raw_subject(rawdata, labels, subject_id)
    processed, excluded = _preprocess_subject(
        rawdata,
        apply_ica=apply_ica,
        seed=seed + subject_id,
    )
    audit = FigshareSubjectAudit(
        subject_id=subject_id,
        archive_member=member,
        ica_enabled=apply_ica,
        ica_excluded_components=excluded,
        sample_rate_hz=OUTPUT_SFREQ,
        marker_codes=_marker_codes(rawdata),
        baseline_samples=(BASELINE_START_SAMPLE, BASELINE_STOP_SAMPLE),
        mi_samples=(MI_START_SAMPLE, MI_STOP_SAMPLE),
    )
    return processed, labels.astype(np.int64) - 1, audit


def _load_mat_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    try:
        variables = loadmat(io.BytesIO(payload), simplify_cells=True)
    except (NotImplementedError, ValueError):
        rawdata, labels = _load_hdf5_payload(payload)
    else:
        rawdata, labels = _extract_scipy_payload(variables)

    return _normalize_raw_axes(rawdata), np.asarray(labels).reshape(-1)


def _extract_scipy_payload(
    variables: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the official eeg struct, with flat-key fixture compatibility."""

    eeg = variables.get("eeg")
    if isinstance(eeg, Mapping):
        rawdata = eeg.get("rawdata")
        labels = eeg.get("label")
        if labels is None:
            labels = eeg.get("labels")
        if rawdata is not None and labels is not None:
            return np.asarray(rawdata), np.asarray(labels)
        eeg_fields = sorted(str(key) for key in eeg)
    else:
        eeg_fields = []

    rawdata = variables.get("rawdata")
    labels = variables.get("label")
    if labels is None:
        labels = variables.get("labels")
    if rawdata is not None and labels is not None:
        return np.asarray(rawdata), np.asarray(labels)

    top_level = sorted(
        str(key) for key in variables if not str(key).startswith("__")
    )
    raise ValueError(
        "Liu2024 MAT file must contain the Figshare 'eeg' struct with "
        "'rawdata' and 'label' fields. Flat rawdata plus label/labels is "
        "accepted only for compatibility. Found top-level variables "
        f"{top_level} and eeg fields {eeg_fields}."
    )


def _load_hdf5_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    try:
        import h5py

        with h5py.File(io.BytesIO(payload), "r") as mat_file:
            container: h5py.Group = mat_file
            eeg_group = mat_file.get("eeg")
            if isinstance(eeg_group, h5py.Group):
                container = eeg_group
            label_key = (
                "label"
                if "label" in container
                else "labels"
                if "labels" in container
                else None
            )
            if "rawdata" not in container or label_key is None:
                raise KeyError(
                    "Expected eeg/rawdata and eeg/label datasets."
                )
            rawdata = np.asarray(container["rawdata"])
            labels = np.asarray(container[label_key])
    except (ImportError, OSError, KeyError, TypeError) as hdf_error:
        raise ValueError(
            "Liu2024 MAT payload could not be decoded as the expected "
            "MATLAB eeg.rawdata/eeg.label schema."
        ) from hdf_error
    return rawdata, labels


def _normalize_raw_axes(rawdata: np.ndarray) -> np.ndarray:
    rawdata = np.asarray(rawdata)
    matches: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        candidate = np.transpose(rawdata, permutation)
        if candidate.shape == RAW_SHAPE:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely normalize rawdata shape {rawdata.shape} to {RAW_SHAPE}."
        )
    return np.asarray(matches[0], dtype=np.float64)


def _marker_codes(rawdata: np.ndarray) -> tuple[int, ...]:
    marker = rawdata[:, 32, :]
    if not np.allclose(marker, np.rint(marker), atol=1e-6):
        return ()
    return tuple(int(value) for value in np.unique(np.rint(marker)))


def _validate_raw_subject(
    rawdata: np.ndarray,
    labels: np.ndarray,
    subject_id: int,
) -> None:
    if rawdata.shape != RAW_SHAPE:
        raise ValueError(
            f"Subject {subject_id} rawdata shape is {rawdata.shape}; expected {RAW_SHAPE}."
        )
    if labels.shape != (RAW_SHAPE[0],):
        raise ValueError(
            f"Subject {subject_id} labels shape is {labels.shape}; expected (40,)."
        )
    expected_labels = np.asarray([1, 2], dtype=labels.dtype)
    if (
        not np.isfinite(labels).all()
        or not np.array_equal(np.unique(labels), expected_labels)
    ):
        raise ValueError(
            f"Subject {subject_id} labels must contain Figshare values 1 and 2."
        )
    label_counts = {
        label: int(np.count_nonzero(labels == label)) for label in (1, 2)
    }
    if label_counts != {1: 20, 2: 20}:
        raise ValueError(
            f"Subject {subject_id} must have 20 trials per class; got {label_counts}."
        )
    if not np.isfinite(rawdata).all():
        raise ValueError(f"Subject {subject_id} rawdata contains non-finite values.")

    marker = rawdata[:, 32, :]
    marker_codes = _marker_codes(rawdata)
    if np.unique(marker).size < 2:
        logger.warning(
            "Subject %s has a constant segmented marker channel; retaining the "
            "documented fixed 2/4/2-second trial timing.",
            subject_id,
        )
    elif not marker_codes:
        logger.warning(
            "Subject %s marker channel is not near-integer; retaining the "
            "documented fixed trial timing.",
            subject_id,
        )
    elif not set(marker_codes).intersection({1, 2, 3}):
        logger.warning(
            "Subject %s marker codes %s do not include documented stages 1/2/3.",
            subject_id,
            marker_codes,
        )

    # CPz is retained as the paper's 30th system channel. Across the official
    # files its acquisition trace can be flat or time-varying and carry a large
    # DC offset, so its raw values are not a valid channel-layout invariant.
    # The 8-30 Hz filter removes the DC level before common-average referencing.


def _preprocess_subject(
    rawdata: np.ndarray,
    *,
    apply_ica: bool,
    seed: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Apply the paper-stated order with explicit reproducible choices."""

    # Work on the complete eight-second trial until the final MI crop. Figshare
    # declares MAT signals in microvolts.
    eeg_eog_uv = np.asarray(rawdata[:, :32, :], dtype=np.float64)
    sos = butter(2, (8.0, 30.0), btype="bandpass", fs=RAW_SFREQ, output="sos")
    filtered_uv = sosfiltfilt(sos, eeg_eog_uv, axis=-1)
    resampled_uv = resample_poly(filtered_uv, up=1, down=2, axis=-1)
    if resampled_uv.shape != (40, 32, 2000):
        raise RuntimeError(
            "Full-trial resampling produced "
            f"{resampled_uv.shape}; expected (40, 32, 2000)."
        )

    # Paper order: filter -> downsample -> common-average reference -> baseline
    # correction -> ICA. CAR reconstructs CPz from the acquisition reference.
    referenced_uv = resampled_uv.copy()
    referenced_uv[:, :30, :] -= referenced_uv[:, :30, :].mean(
        axis=1,
        keepdims=True,
    )
    baseline = referenced_uv[
        :, :, BASELINE_START_SAMPLE:BASELINE_STOP_SAMPLE
    ].mean(axis=-1, keepdims=True)
    baseline_corrected_uv = referenced_uv - baseline

    excluded: tuple[int, ...] = ()
    if apply_ica:
        info = mne.create_info(
            ch_names=list(RAW_DATA_CHANNEL_NAMES[:32]),
            sfreq=OUTPUT_SFREQ,
            ch_types="eeg",
        )
        info.set_channel_types({name: "eog" for name in EOG_NAMES})
        epochs = mne.EpochsArray(
            baseline_corrected_uv * 1e-6,
            info,
            tmin=-2.0,
            baseline=None,
            verbose="ERROR",
        )
        ica = ICA(
            n_components=0.99,
            method="fastica",
            random_state=seed,
            max_iter="auto",
        )
        ica.fit(
            epochs,
            picks=list(range(30)),
            reject_by_annotation=True,
            verbose="ERROR",
        )
        detected: set[int] = set()
        for eog_name in EOG_NAMES:
            indices, _ = ica.find_bads_eog(
                epochs,
                ch_name=eog_name,
                threshold=3.0,
                verbose="ERROR",
            )
            detected.update(int(index) for index in indices)
        excluded = tuple(sorted(detected))
        ica.apply(epochs, exclude=list(excluded), verbose="ERROR")
        cleaned_uv = epochs.get_data(copy=True) * 1e6
    else:
        cleaned_uv = baseline_corrected_uv

    mi_uv = cleaned_uv[
        :, :30, MI_START_SAMPLE:MI_STOP_SAMPLE
    ]

    expected_shape = (40, 30, 1000)
    if mi_uv.shape != expected_shape:
        raise RuntimeError(
            f"Paper-aligned preprocessing produced {mi_uv.shape}; "
            f"expected {expected_shape}."
        )
    if not np.isfinite(mi_uv).all():
        raise ValueError("Paper-aligned preprocessing produced non-finite values.")
    return np.asarray(mi_uv, dtype=np.float32), excluded


def _validate_archive_identity(path: Path) -> None:
    size = path.stat().st_size
    if size != FIGSHARE_RAW_SIZE:
        raise ValueError(
            f"Unexpected sourcedata.zip size {size}; expected {FIGSHARE_RAW_SIZE}."
        )
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != FIGSHARE_RAW_MD5:
        raise ValueError(
            f"Checksum mismatch for {path}; expected MD5 {FIGSHARE_RAW_MD5}."
        )


def _validate_participants_identity(path: Path) -> None:
    size = path.stat().st_size
    if size != FIGSHARE_PARTICIPANTS_SIZE:
        raise ValueError(
            f"Unexpected participants.tsv size {size}; "
            f"expected {FIGSHARE_PARTICIPANTS_SIZE}."
        )
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if digest != FIGSHARE_PARTICIPANTS_MD5:
        raise ValueError(
            f"Checksum mismatch for {path}; "
            f"expected MD5 {FIGSHARE_PARTICIPANTS_MD5}."
        )
