"""Run the within-subject cross-validation and save the result as JSON.

    python -m src.ourexpv24liu.train.run_within --data-dir /path/to/MNE-liu2024-data --output result.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ..data import DATASETS, dataset_config, subject_file
from .within import WITHIN_DEFAULTS, run_within, within_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="liu2024", help="Default: liu2024")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="liu2024: the MNE-liu2024-data folder; bci2a: the folder with A01T.mat ... A09T.mat")
    parser.add_argument("--output", type=Path, help="Result JSON (default: timestamped file in ./outputs)")
    parser.add_argument("--subjects", type=int, nargs="+", help="Subjects to run (default: all configured)")
    parser.add_argument("--all-subjects", action="store_true",
                        help="Also use the subjects left out by default: the three duplicated recordings of liu2024 "
                             "(5, 15, 49), or subject 4 of bci2a")
    parser.add_argument("--exclude-subjects", type=int, nargs="+", default=(), metavar="SUBJECT",
                        help="Leave these subjects out of the experiment entirely (training and testing)")
    parser.add_argument("--window", type=float, nargs=2, metavar=("START", "STOP"),
                        help="Analysed window in seconds from the imagery onset (default: 0 4 for liu2024, 0 2 for bci2a)")
    parser.add_argument("--folds", type=int, default=10, help="Folds per subject (default: 10)")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed of the fold split (default: 42)")
    parser.add_argument("--epochs", type=int, help=f"Epochs per model (default: {WITHIN_DEFAULTS['epochs']})")
    parser.add_argument("--seed", type=int, help="Seed of the first model of each fold (default: 12345)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--quiet", action="store_true", help="Hide the per-model progress bars")
    args = parser.parse_args(argv)

    options = dict(all_subjects=True) if args.all_subjects else {}
    if args.window:
        options["window_seconds"] = tuple(args.window)
    try:
        data_config = dataset_config(args.dataset, **options)
        unknown = sorted(set(args.exclude_subjects) - set(data_config.subjects))
        if unknown:
            raise ValueError(f"Cannot exclude subjects that are not in the experiment: {unknown}")
        kept = tuple(s for s in data_config.subjects if s not in args.exclude_subjects)
        data_config = dataset_config(args.dataset, **options, subjects=kept)
    except ValueError as error:
        parser.error(str(error))
    overrides = {name: getattr(args, name) for name in ("epochs", "seed", "device") if getattr(args, name) is not None}
    config = within_config(**overrides, show_progress=not args.quiet)
    subjects = args.subjects or data_config.subjects
    missing = [subject_file(args.data_dir, s, data_config).name for s in subjects if not subject_file(args.data_dir, s, data_config).is_file()]
    if missing:
        parser.error(f"Missing recordings in {args.data_dir}: {', '.join(missing)}")
    try:
        result = run_within(args.data_dir, data_config, config=config, subjects=args.subjects,
                            folds=args.folds, split_seed=args.split_seed)
    except ValueError as error:
        parser.error(str(error))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("outputs") / f"Version24Liu_{args.dataset}_Within_seed{config.seed}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Result saved to {output}")


if __name__ == "__main__":
    main()
