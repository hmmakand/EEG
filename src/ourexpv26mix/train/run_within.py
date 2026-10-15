"""Run the within-subject cross-validation and save the result as JSON.

    python -m src.ourexpv26mix.train.run_within --data-dir /path/to/MNE-liu2024-data --output result.json
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ..data import DATASETS, dataset_config, subject_file
from ..model import ModelConfig
from .within import WITHIN_DEFAULTS, run_within, within_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="liu2024", help="Default: liu2024")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="liu2024: the MNE-liu2024-data folder; jia2019: the folder with exp1-S1-left.mat ...; bci2a: the folder with A01T.mat ... A09T.mat")
    parser.add_argument("--output", type=Path, help="Result JSON (default: timestamped file in ./outputs)")
    parser.add_argument("--subjects", type=int, nargs="+", help="Subjects to run (default: all configured)")
    parser.add_argument("--all-subjects", action="store_true",
                        help="Also use the subjects left out by default: the three duplicated recordings of liu2024 "
                             "(5, 15, 49), or subject 4 of bci2a")
    parser.add_argument("--node-time-attention", type=int, metavar="BLOCKS",
                        help="Transformer blocks over the time steps of every node's signal features (default: 0)")
    parser.add_argument("--layers", type=int, help="Number of hybrid blocks in the graph head (default: 2)")
    parser.add_argument("--readout", choices=("mean", "electrode"),
                        help="How the graph head reads out the nodes (default: mean)")
    parser.add_argument("--node-signal", choices=("off", "power", "timed", "mixed"),
                        help="Give the graph nodes features learned from the electrode's signal (default: off)")
    parser.add_argument("--paths", nargs="+", choices=("graph", "wide", "temporal"),
                        help="Ablation: use only these paths of the model (default: all three)")
    parser.add_argument("--use-subjects", type=int, nargs="+", metavar="SUBJECT",
                        help="Run the experiment on exactly these subjects (training and testing), "
                             "instead of the dataset's default list")
    parser.add_argument("--shuffle-labels", action="store_true",
                        help="Control run: permute the training labels, so that nothing can be learned")
    parser.add_argument("--exclude-subjects", type=int, nargs="+", default=(), metavar="SUBJECT",
                        help="Leave these subjects out of the experiment entirely (training and testing)")
    parser.add_argument("--task", choices=("left-right", "imagery-rest"),
                        help="What is classified. imagery-rest: imagery against the break after it, the default "
                             "for liu2024. left-right: the imagined hand, the default for bci2a and jia2019")
    parser.add_argument("--tangent-features", action="store_true",
                        help="Add the logarithm of the trial's covariance (tangent space) to the node and edge features")
    parser.add_argument("--low-frequency", action="store_true",
                        help="Add a path that classifies from the 0.5-4 Hz potentials. Not valid for a motor-imagery "
                             "result on bci2a: there this band reflects eye movements toward the arrow cue (README, 10)")
    parser.add_argument("--window", type=float, nargs=2, metavar=("START", "STOP"),
                        help="Analysed window in seconds from the imagery onset (default: 0 1.9 for liu2024 imagery-rest, 0 4 for liu2024 left-right, 0 3.5 for jia2019, 0 2 for bci2a)")
    parser.add_argument("--folds", type=int, default=10, help="Folds per subject (default: 10)")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed of the fold split (default: 42)")
    parser.add_argument("--epochs", type=int, help=f"Epochs per model (default: {WITHIN_DEFAULTS['epochs']})")
    parser.add_argument("--seed", type=int, help="Seed of the first model of each fold (default: 12345)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--quiet", action="store_true", help="Hide the per-model progress bars")
    args = parser.parse_args(argv)

    if args.all_subjects and args.dataset == "jia2019":
        parser.error("--all-subjects does not apply to jia2019; all 15 subjects are used")
    options = dict(all_subjects=True) if args.all_subjects else {}
    if args.task:
        options["task"] = args.task.replace("-", "_")
    if args.tangent_features:
        options["tangent_features"] = True
    if args.low_frequency:
        options["low_band"] = (0.5, 4.0)
    if args.window:
        options["window_seconds"] = tuple(args.window)
    try:
        data_config = dataset_config(args.dataset, **options)
        if args.use_subjects:
            data_config = dataset_config(args.dataset, **options, subjects=tuple(args.use_subjects))
        unknown = sorted(set(args.exclude_subjects) - set(data_config.subjects))
        if unknown:
            raise ValueError(f"Cannot exclude subjects that are not in the experiment: {unknown}")
        kept = tuple(s for s in data_config.subjects if s not in args.exclude_subjects)
        data_config = dataset_config(args.dataset, **options, subjects=kept)
    except ValueError as error:
        parser.error(str(error))
    overrides = {name: getattr(args, name) for name in ("epochs", "seed", "device") if getattr(args, name) is not None}
    config = within_config(**overrides, show_progress=not args.quiet,
        shuffle_training_labels=args.shuffle_labels)
    subjects = args.subjects or data_config.subjects
    missing = [subject_file(args.data_dir, s, data_config).name for s in subjects if not subject_file(args.data_dir, s, data_config).is_file()]
    if missing:
        parser.error(f"Missing recordings in {args.data_dir}: {', '.join(missing)}")
    try:
        model_options = dict(paths=tuple(args.paths)) if args.paths else {}
        for name in ("node_signal", "readout", "layers", "node_time_attention"):
            if getattr(args, name):
                model_options[name] = getattr(args, name)
        model_config = ModelConfig.for_data(data_config, **model_options)
        result = run_within(args.data_dir, data_config, model_config, config, subjects=args.subjects,
                            folds=args.folds, split_seed=args.split_seed)
    except ValueError as error:
        parser.error(str(error))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("outputs") / f"Version26Mix_{args.dataset}_Within_seed{config.seed}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Result saved to {output}")


if __name__ == "__main__":
    main()
