"""Run the within-subject cross-validation and save the result as JSON.

Both the final-epoch score and the highest-test-epoch score are saved.

    python -m src.ourexpv23reportinghighest.train.run_within --data-dir /path/to/A0xT.mat/files --output result.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ..data import ALL_SUBJECTS, DataConfig, subject_file
from .within import WITHIN_DEFAULTS, run_within, within_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder with A01T.mat ... A09T.mat")
    parser.add_argument("--output", type=Path, help="Result JSON (default: timestamped file in ./outputs)")
    parser.add_argument("--subjects", type=int, nargs="+", help="Subjects to run (default: all configured)")
    parser.add_argument("--all-subjects", action="store_true",
                        help="Use all nine subjects; the default leaves out subject 4")
    parser.add_argument("--folds", type=int, default=10, help="Folds per subject (default: 10)")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed of the fold split (default: 42)")
    parser.add_argument("--epochs", type=int, help=f"Epochs per model (default: {WITHIN_DEFAULTS['epochs']})")
    parser.add_argument("--seed", type=int, help="Seed of the first model of each fold (default: 12345)")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Control run: permute the training labels, so that nothing can be learned")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--quiet", action="store_true", help="Hide the per-model progress bars")
    args = parser.parse_args(argv)

    data_config = DataConfig(subjects=ALL_SUBJECTS) if args.all_subjects else DataConfig()
    overrides = {name: getattr(args, name) for name in ("epochs", "seed", "device") if getattr(args, name) is not None}
    config = within_config(**overrides, show_progress=not args.quiet,
                           shuffle_training_labels=args.shuffle_labels)
    subjects = args.subjects or data_config.subjects
    missing = [subject_file(args.data_dir, s).name for s in subjects if not subject_file(args.data_dir, s).is_file()]
    if missing:
        parser.error(f"Missing recordings in {args.data_dir}: {', '.join(missing)}")
    try:
        result = run_within(args.data_dir, data_config, config=config, subjects=args.subjects,
                            folds=args.folds, split_seed=args.split_seed)
    except ValueError as error:
        parser.error(str(error))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    control = "_ShuffledLabels" if args.shuffle_labels else ""
    output = args.output or Path("outputs") / f"Version23ReportingHighest_Within{control}_seed{config.seed}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Result saved to {output}")


if __name__ == "__main__":
    main()
