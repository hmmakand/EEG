"""Run the leave-one-subject-out experiment and save the result as JSON.

    python -m src.ourexpv26mix.train.run_loso --data-dir /path/to/MNE-liu2024-data --output result.json
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from ..data import DATASETS, dataset_config, subject_file
from ..model import ModelConfig
from .config import TrainConfig
from .loso import run_loso


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="liu2024", help="Default: liu2024")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="liu2024: the MNE-liu2024-data folder; jia2019: the folder with exp1-S1-left.mat ...; bci2a: the folder with A01T.mat ... A09T.mat")
    parser.add_argument("--output", type=Path, help="Result JSON (default: timestamped file in ./outputs)")
    parser.add_argument("--subjects", type=int, nargs="+", help="Held-out subjects to run (default: all configured)")
    parser.add_argument("--all-subjects", action="store_true",
                        help="Also use the subjects left out by default: the three duplicated recordings of liu2024 "
                             "(5, 15, 49), or subject 4 of bci2a")
    parser.add_argument("--target-calibration", action=argparse.BooleanOptionalAction,
                        help="After a warm-up, also train on the unlabelled test subject with the model's own "
                             "confident, physiologically checked predictions as targets (PGTC). On by default; "
                             "--no-target-calibration gives the training of version 23")
    parser.add_argument("--calibration-warmup-epochs", type=int, help="Default: half of the epochs")
    parser.add_argument("--calibration-source-weight", type=float, help="Default: 0.95")
    parser.add_argument("--calibration-confidence", type=float, help="Default: 0.90")
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
    parser.add_argument("--tangent-features", action="store_true",
                        help="Add the logarithm of the trial's covariance (tangent space) to the node and edge features")
    parser.add_argument("--low-frequency", action="store_true",
                        help="Add a path that classifies from the 0.5-4 Hz potentials")
    parser.add_argument("--window", type=float, nargs=2, metavar=("START", "STOP"),
                        help="Analysed window in seconds from the imagery onset (default: 0 4 for liu2024, 0 3.5 for jia2019, 0 2 for bci2a)")
    parser.add_argument("--epochs", type=int, help="Epochs per model (default: 20)")
    parser.add_argument("--seed", type=int, help="Seed of the first model of each fold (default: 12345)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--quiet", action="store_true", help="Hide the per-model progress bars")
    args = parser.parse_args(argv)

    if args.all_subjects and args.dataset == "jia2019":
        parser.error("--all-subjects does not apply to jia2019; all 15 subjects are used")
    options = dict(all_subjects=True) if args.all_subjects else {}
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
    overrides = {name: getattr(args, name) for name in ("epochs", "seed", "device", "calibration_warmup_epochs", "calibration_source_weight",
                                                   "calibration_confidence") if getattr(args, name) is not None}
    config = replace(TrainConfig(), **overrides, show_progress=not args.quiet,
        shuffle_training_labels=args.shuffle_labels,
        **({} if args.target_calibration is None else dict(target_calibration=args.target_calibration)))
    missing = [subject_file(args.data_dir, s, data_config).name for s in data_config.subjects
               if not subject_file(args.data_dir, s, data_config).is_file()]
    if missing:
        parser.error(f"Missing recordings in {args.data_dir}: {', '.join(missing)}")
    try:
        model_options = dict(paths=tuple(args.paths)) if args.paths else {}
        for name in ("node_signal", "readout", "layers", "node_time_attention"):
            if getattr(args, name):
                model_options[name] = getattr(args, name)
        model_config = ModelConfig.for_data(data_config, **model_options)
        result = run_loso(args.data_dir, data_config, model_config, config, held_out_subjects=args.subjects)
    except ValueError as error:
        parser.error(str(error))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("outputs") / f"Version26Mix_{args.dataset}_LOSO_seed{config.seed}_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Result saved to {output}")


if __name__ == "__main__":
    main()
