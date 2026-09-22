"""Run PLV graph training with FixMatch pseudo-labeling on labeled training folds."""
from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path

if __package__ in (None, "", "train"):
    # Support direct execution and copying this experiment outside the repository.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import default_dataset_config
    from dataset.data_io import resolve_data_dir
    from train.config import default_training_config, validate_training_config
    from train.experiment import run_experiment
    from train.reporting import print_summary, save_results, resolve_results_path
    from train.setup import select_device
    from train.train import validate_input_files
else:
    from ..dataset import default_dataset_config
    from ..dataset.data_io import resolve_data_dir
    from .config import default_training_config, validate_training_config
    from .experiment import run_experiment
    from .reporting import print_summary, save_results, resolve_results_path
    from .setup import select_device
    from .train import validate_input_files


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Directory containing A01T.mat, etc.")
    parser.add_argument("--output", type=Path,
                        help="Results JSON path (default filename includes PLV_FIXMATCH and graph threshold)")
    parser.add_argument("--fixmatch-lambda-u", type=float, default=None)
    parser.add_argument("--fixmatch-confidence-threshold", type=float, default=None)
    parser.add_argument("--fixmatch-weak-noise-std", type=float, default=None)
    parser.add_argument("--fixmatch-strong-noise-std", type=float, default=None)
    parser.add_argument("--fixmatch-strong-mask-prob", type=float, default=None)
    parser.add_argument("--fixmatch-reuse-labeled", action="store_true", default=None,
                        dest="fixmatch_reuse_labeled_as_unlabeled",
                        help="Feed each fold's own labeled training graphs through the "
                             "FixMatch unlabeled path too (labels dropped for that pass)")
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=None)
    parser.add_argument("--folds", dest="n_folds", type=int, default=None)
    parser.add_argument("--subjects", type=int, nargs="+", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dataset_config = default_dataset_config()
    overrides = {name: getattr(args, name) for name in (
        "fixmatch_lambda_u", "fixmatch_confidence_threshold", "fixmatch_weak_noise_std",
        "fixmatch_strong_noise_std", "fixmatch_strong_mask_prob", "fixmatch_reuse_labeled_as_unlabeled",
        "num_epochs", "n_folds", "device",
    ) if getattr(args, name) is not None}
    if args.subjects is not None:
        overrides["subjects"] = tuple(args.subjects)
    training_config = replace(default_training_config(), training_engine="fixmatch", **overrides)
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
        f"{dataset_config.connectivity.method}_FIXMATCH",
        args.output if args.output is not None else training_config.output_path,
        threshold=dataset_config.graphs.threshold,
    )
    save_results(subject_results, output_path)


if __name__ == "__main__":
    main()
