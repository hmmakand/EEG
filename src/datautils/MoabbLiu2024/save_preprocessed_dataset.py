"""Preprocess and save all Liu2024 subjects for efficient PyTorch access."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

if __package__:
    from .config import DEFAULT_DATA_ROOT, Liu2024Config
    from .dataset import prepare_liu2024_dataset
else:
    # Support ``python src/datautils/MoabbLiu2024/save_preprocessed_dataset.py``
    # in addition to the preferred ``python -m ...`` package invocation.
    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository_root))
    from src.datautils.MoabbLiu2024.config import DEFAULT_DATA_ROOT, Liu2024Config
    from src.datautils.MoabbLiu2024.dataset import prepare_liu2024_dataset


ALL_SUBJECTS = tuple(range(1, 51))
# DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "Preprocessed-MNE-liu2024-data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "Preprocessed-Gamma-31-40-MNE-liu2024-data"
EXPECTED_TRIALS_PER_SUBJECT = 40


def _metadata(
    config: Liu2024Config,
    channel_names: list[str],
    window_shape: tuple[int, ...],
) -> dict:
    return {
        "dataset": "Liu2024",
        "format_version": 1,
        "array_format": "NumPy .npy",
        "subjects": list(config.subject_ids),
        "excluded_subjects": [],
        "class_mapping": config.event_mapping,
        "n_windows": len(config.subject_ids) * EXPECTED_TRIALS_PER_SUBJECT,
        "window_shape": list(window_shape),
        "signal_dtype": "float32",
        "target_dtype": "int64",
        "sampling_frequency_hz": config.target_sfreq,
        "channel_names": channel_names,
        "preprocessing": {
            "bandpass_hz": [config.l_freq, config.h_freq],
            "standardize": config.standardize,
            "standardize_factor_new": config.standardize_factor_new,
            "standardize_init_block_seconds": config.standardize_init_block_seconds,
            "standardize_init_block_size": config.standardize_init_block_size,
            "trial_start_offset_seconds": config.trial_start_offset_seconds,
            "trial_stop_offset_seconds": config.trial_stop_offset_seconds,
        },
    }


def save_preprocessed_dataset(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    subject_ids: tuple[int, ...] = ALL_SUBJECTS,
    overwrite: bool = False,
) -> Path:
    """Prepare subjects individually and atomically save the resulting arrays."""

    output = Path(output_dir).expanduser().resolve()
    if not subject_ids:
        raise ValueError("subject_ids cannot be empty")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))

    total_windows = len(subject_ids) * EXPECTED_TRIALS_PER_SUBJECT
    X = y = saved_subjects = trial_indices = None
    cursor = 0
    channel_names: list[str] | None = None
    window_shape: tuple[int, ...] | None = None
    try:
        for subject in subject_ids:
            config = Liu2024Config(subject_ids=(subject,), preload_windows=True)
            windows = prepare_liu2024_dataset(config)
            if len(windows) != EXPECTED_TRIALS_PER_SUBJECT:
                raise ValueError(
                    f"Subject {subject} has {len(windows)} windows; expected "
                    f"{EXPECTED_TRIALS_PER_SUBJECT}"
                )

            current_channels = list(windows.datasets[0].raw.ch_names)
            if channel_names is None:
                channel_names = current_channels
                first_x = np.asarray(windows[0][0])
                window_shape = tuple(first_x.shape)
                X = open_memmap(
                    temporary / "X.npy", mode="w+", dtype=np.float32,
                    shape=(total_windows, *window_shape),
                )
                y = open_memmap(
                    temporary / "y.npy", mode="w+", dtype=np.int64,
                    shape=(total_windows,),
                )
                saved_subjects = open_memmap(
                    temporary / "subject_ids.npy", mode="w+", dtype=np.int64,
                    shape=(total_windows,),
                )
                trial_indices = open_memmap(
                    temporary / "trial_indices.npy", mode="w+", dtype=np.int64,
                    shape=(total_windows,),
                )
            elif current_channels != channel_names:
                raise ValueError(f"Subject {subject} has inconsistent EEG channels")

            # The first subject allocates every memory map. Keeping this
            # invariant explicit also lets static type checkers narrow away
            # the initial None values below.
            assert X is not None
            assert y is not None
            assert saved_subjects is not None
            assert trial_indices is not None
            for trial in range(len(windows)):
                x, target, _ = windows[trial]
                X[cursor] = np.asarray(x, dtype=np.float32)
                y[cursor] = int(target)
                saved_subjects[cursor] = subject
                trial_indices[cursor] = trial
                cursor += 1
            print(f"Saved subject {subject}: {cursor}/{total_windows} windows", flush=True)

        if cursor != total_windows or channel_names is None or window_shape is None:
            raise RuntimeError(f"Saved {cursor} windows; expected {total_windows}")
        assert X is not None
        assert y is not None
        assert saved_subjects is not None
        assert trial_indices is not None
        for array in (X, y, saved_subjects, trial_indices):
            array.flush()

        counts = Counter(map(int, np.asarray(y)))
        expected_per_class = total_windows // 2
        if counts != Counter({0: expected_per_class, 1: expected_per_class}):
            raise ValueError(f"Unexpected class counts: {dict(counts)}")
        if not np.isfinite(np.asarray(X)).all():
            raise ValueError("Saved EEG data contain non-finite values")

        config = Liu2024Config(subject_ids=subject_ids)
        metadata = _metadata(config, channel_names, window_shape)
        with (temporary / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")

        # Release memory maps before moving/replacing their directory.
        del X, y, saved_subjects, trial_indices
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(f"Dataset saved to {output}")
    print(f"Windows: {total_windows}; class counts: {{0: {total_windows // 2}, 1: {total_windows // 2}}}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output directory"
    )
    args = parser.parse_args()
    save_preprocessed_dataset(args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
