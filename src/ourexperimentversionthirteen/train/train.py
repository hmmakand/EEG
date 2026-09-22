"""Run the standalone version thirteen supervised BCI Competition IV 2a experiment (per-subject CV)."""
from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path

if __package__ in (None, "", "train"):
    # Support direct execution and copying this experiment outside the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import default_dataset_config
    from dataset.data_io import resolve_data_dir, subject_file_path
    from train.config import default_training_config, validate_training_config
    from train.experiment import run_experiment
    from train.reporting import print_summary, save_results, resolve_results_path
    from train.setup import select_device
else:
    from ..dataset import default_dataset_config
    from ..dataset.data_io import resolve_data_dir, subject_file_path
    from .config import default_training_config, validate_training_config
    from .experiment import run_experiment
    from .reporting import print_summary, save_results, resolve_results_path
    from .setup import select_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Directory containing A01T.mat, etc.")
    parser.add_argument("--output", type=Path,
                        help="Results JSON path (default filename includes the connectivity method and graph threshold)")
    parser.add_argument("--epochs", dest="num_epochs", type=int, help="Epochs per fold (default: 249)")
    parser.add_argument("--folds", dest="n_folds", type=int, help="Folds per subject (default: 10)")
    parser.add_argument("--subjects", type=int, nargs="+", help="Subject IDs (default: 1 2 3 5 6 7 8 9)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="Compute device (default: auto)")
    return parser.parse_args(argv)


def validate_input_files(data_dir, subjects) -> None:
    missing = [subject_file_path(data_dir, subject).name for subject in subjects
               if not subject_file_path(data_dir, subject).is_file()]
    if missing:
        raise ValueError(f"Missing data files in {data_dir}: {', '.join(missing)}. "
                         "Use --data-dir to select the dataset directory.")


def main(argv=None):
    args = parse_args(argv)
    dataset_config = default_dataset_config()
    overrides = {name: getattr(args, name) for name in ("num_epochs", "n_folds", "device")
                 if getattr(args, name) is not None}
    if args.subjects is not None:
        overrides["subjects"] = tuple(args.subjects)
    training_config = replace(default_training_config(), **overrides)
    data_dir = resolve_data_dir(args.data_dir)
    try:
        validate_training_config(training_config)
        validate_input_files(data_dir, training_config.subjects)
        device = select_device(training_config.device)
    except ValueError as error:
        # Preserve argparse-style failures for invalid input without starting training.
        argparse.ArgumentParser(description=__doc__).error(str(error))
    print(f"Using device: {device}")
    subject_results = run_experiment(data_dir, device, dataset_config, training_config)
    print_summary(subject_results)
    output_path = resolve_results_path(
        dataset_config.connectivity.method,
        args.output if args.output is not None else training_config.output_path,
        threshold=dataset_config.graphs.threshold,
    )
    save_results(subject_results, output_path)


if __name__ == "__main__":
    main()
