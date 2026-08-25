"""Command-line verification for the prepared Liu2024 dataset."""

import argparse
from collections import Counter

import numpy as np

from .config import Liu2024Config
from .dataset import prepare_liu2024_dataset


def verify(subject_ids: tuple[int, ...]) -> None:
    """Prepare subjects and print structural and numerical checks."""

    config = Liu2024Config(subject_ids=subject_ids)
    windows = prepare_liu2024_dataset(config)
    targets = Counter()
    shapes = set()
    all_finite = True
    for index in range(len(windows)):
        x, y, _ = windows[index]
        shapes.add(tuple(x.shape))
        targets[int(y)] += 1
        all_finite = all_finite and bool(np.isfinite(x).all())

    first_raw = windows.datasets[0].raw
    print(f"Subjects requested: {len(subject_ids)}")
    print(f"Recordings: {len(windows.datasets)}")
    print(f"Trials: {len(windows)}")
    print(f"Class counts (0=left, 1=right): {dict(sorted(targets.items()))}")
    print(f"Trial shapes: {sorted(shapes)}")
    print(f"Sampling frequency: {first_raw.info['sfreq']} Hz")
    print(f"EEG channels: {len(first_raw.ch_names)}")
    print(f"All values finite: {all_finite}")
    if len(shapes) != 1 or not all_finite or set(targets) != {0, 1}:
        raise RuntimeError("Liu2024 verification failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="+",
        default=[1],
        help="Subject IDs to verify (default: 1). Use 1 2 ... 50 for all subjects.",
    )
    args = parser.parse_args()
    verify(tuple(args.subjects))


if __name__ == "__main__":
    main()
