"""Path helpers for dataset cache locations.

Hydra changes the process working directory for each run, so relative dataset
paths should be resolved against the original project directory. Real dataset
configs should normally provide `data_dir`; when they do not, the conventional
MNE cache directory `~/mne_data` is used as a fallback.
"""

from __future__ import annotations

from pathlib import Path

from hydra.utils import get_original_cwd


def resolve_data_dir(value: str | None) -> Path:
    """Resolve a configured dataset cache path to an absolute path.

    Absolute paths are returned unchanged. Relative paths are anchored to
    Hydra's original working directory, which keeps `data/moabb` inside the
    project even when a run executes under `outputs/...`.
    """

    if value is None:
        return Path.home() / "mne_data"

    data_dir = Path(str(value)).expanduser()
    if data_dir.is_absolute():
        return data_dir

    try:
        base_dir = Path(get_original_cwd())
    except ValueError:
        base_dir = Path.cwd()
    return base_dir / data_dir
