"""Run the within-subject cross-validation and save the result as JSON.

    python -m src.ourexpv27.train.run_within --data-dir /path/to/MNE-liu2024-data --output-dir outputs
    python -m src.ourexpv27.train.run_within --dataset bci2a --data-dir /path/to/A0xT.mat/files --output-dir outputs

The result goes to <output-dir>/<dataset>/.
"""
import argparse

from ..datascripts import dataset_config
from .output import models_folder, result_path, write_result
from .run_loso import add_common_arguments, check_recordings, variant_name
from .within import WITHIN_DEFAULTS, run_within, within_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--folds", type=int, default=10, help="Folds per subject (default: 10)")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed of the fold split (default: 42)")
    parser.add_argument("--epochs", type=int, help=f"Epochs per model (default: {WITHIN_DEFAULTS['epochs']})")
    args = parser.parse_args(argv)

    data_config = dataset_config(args.dataset)
    if args.dataset == "jia2019":
        print("Warning: within-subject results on jia2019 are not valid. The two classes are stored in separate "
              "files, and a classifier recognises the file (README, section 4.3).", flush=True)
    overrides = {name: getattr(args, name) for name in ("epochs", "seed", "device") if getattr(args, name) is not None}
    config = within_config(**overrides, show_progress=not args.quiet)
    check_recordings(parser, args, data_config, args.subjects or data_config.subjects)
    output = args.output or result_path(args.output_dir, args.dataset, "Within", config.seed,
                                        variant_name(args.paths))
    try:
        result = run_within(args.data_dir, data_config, config, subjects=args.subjects, folds=args.folds,
                            split_seed=args.split_seed, models_dir=models_folder(output) if args.save_models else None,
                            paths=args.paths)
    except ValueError as error:
        parser.error(str(error))
    write_result(result, output)


if __name__ == "__main__":
    main()
