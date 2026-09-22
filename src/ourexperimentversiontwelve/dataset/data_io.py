"""Filesystem and MATLAB decoding, independent of trial selection."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.io as sio


@dataclass(frozen=True)
class Session:
    signal: np.ndarray
    positions: np.ndarray
    labels: np.ndarray
    session_id: int


def resolve_data_dir(data_dir: str | Path | None = None) -> str:
    """Resolve data independently of the working directory or Git metadata."""
    if data_dir is not None:
        return str(Path(data_dir).expanduser().resolve())

    experiment_dir = Path(__file__).resolve().parents[1]
    dataset_name = "bci-competition-iv-data-sets-2a"
    candidates = [
        experiment_dir / "data" / dataset_name,
        Path("/kaggle/input") / dataset_name,
    ]
    # Preserve the existing repository data location when running in src/.
    if experiment_dir.parent.name == "src":
        candidates.append(experiment_dir.parent.parent / "data" / "kaggle" / dataset_name)
    return str(next((path for path in candidates if path.is_dir()), candidates[0]))



def subject_file_path(data_dir, subject_number: int) -> Path:
    return Path(data_dir) / f"A{subject_number:02d}T.mat"


def load_mat_file(path):
    return sio.loadmat(path)["data"]


def read_sessions(data):
    for index in range(data.shape[1]):
        record = data[0, index][0, 0]
        yield Session(record[0], record[1].flatten(), record[2].flatten(), index)
