"""Load Liu2024 from the repository's local MOABB cache."""

import os
from pathlib import Path
from typing import Iterable

from braindecode.datasets import MOABBDataset

from .config import DEFAULT_DATA_ROOT


def _normalise_subject_ids(subject_ids: Iterable[int] | None) -> list[int]:
    subjects = list(range(1, 51)) if subject_ids is None else [int(s) for s in subject_ids]
    if not subjects:
        raise ValueError("subject_ids cannot be empty")
    if len(subjects) != len(set(subjects)):
        raise ValueError("subject_ids contains duplicates")
    invalid = [subject for subject in subjects if subject < 1 or subject > 50]
    if invalid:
        raise ValueError(f"Liu2024 subject IDs must be between 1 and 50; got {invalid}")
    return subjects


def _check_local_files(data_root: Path, subject_ids: list[int]) -> None:
    dataset_dir = data_root / "MNE-liu2024-data" / "files"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Local Liu2024 dataset was not found at {dataset_dir}")

    missing = []
    for subject in subject_ids:
        filename = f"sub-{subject:02d}_task-motor-imagery_eeg.edf"
        path = dataset_dir / "edffile" / f"sub-{subject:02d}" / "eeg" / filename
        if not path.is_file():
            missing.append(str(path))
    if missing:
        preview = "\n".join(missing[:5])
        suffix = "\n..." if len(missing) > 5 else ""
        raise FileNotFoundError(f"Missing Liu2024 EDF file(s):\n{preview}{suffix}")


def load_liu2024(
    subject_ids: Iterable[int] | None = None,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> MOABBDataset:
    """Load requested Liu2024 subjects through MOABB without downloading data."""

    subjects = _normalise_subject_ids(subject_ids)
    root = Path(data_root).expanduser().resolve()
    _check_local_files(root, subjects)
    os.environ["MNE_DATASETS_LIU2024_PATH"] = str(root)
    return MOABBDataset(dataset_name="Liu2024", subject_ids=subjects)
