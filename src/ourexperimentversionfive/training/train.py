"""Command-line entry point for weighted-GCN LOSO training on a selectable combination."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ourexperimentversionfive.data.combinations import COMBINATIONS
from src.ourexperimentversionfive.training.config import TrainingConfig
from src.ourexperimentversionfive.training.loso import (
    train_all_loso_folds,
    train_loso_fold,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed-dataset training command-line interface."""

    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-normalization", choices=("none", "zscore"), default="none")
    parser.add_argument(
        "--combination",
        choices=sorted(COMBINATIONS),
        default=defaults.combination,
        help="which node/edge/band feature combination to train on",
    )
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument("--test-subject", type=int)
    subject_group.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=defaults.learning_rate
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults.weight_decay
    )
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "adam", "sgd"),
        default=defaults.optimizer,
        help="which optimizer to train with",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=defaults.momentum,
        help="SGD momentum; ignored for adamw/adam",
    )
    patience_group = parser.add_mutually_exclusive_group()
    patience_group.add_argument(
        "--patience", type=int, default=defaults.patience
    )
    patience_group.add_argument(
        "--no-early-stopping",
        action="store_const",
        const=None,
        dest="patience",
        help="disable early stopping and always run the full --epochs budget",
    )
    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=defaults.minimum_improvement,
    )
    clipping_group = parser.add_mutually_exclusive_group()
    clipping_group.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=defaults.gradient_clip_norm,
    )
    clipping_group.add_argument(
        "--no-gradient-clipping",
        action="store_const",
        const=None,
        dest="gradient_clip_norm",
    )
    parser.add_argument(
        "--validation-subjects",
        type=int,
        default=defaults.validation_subjects,
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--seed-strategy",
        choices=("shared", "per_fold"),
        default=defaults.seed_strategy,
        help="reuse one initialization seed or derive it from each test subject",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--num-workers", type=int, default=defaults.num_workers
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--run-name",
        help="unique run directory name; defaults to a UTC timestamp",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow writing into an existing non-empty run directory",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="disable all output artifacts"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse options, run the requested fold scope, and print JSON to stdout."""

    args = build_parser().parse_args(argv)
    options = {
        "combination": args.combination,
        "node_normalization": args.node_normalization,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "patience": args.patience,
        "minimum_improvement": args.minimum_improvement,
        "gradient_clip_norm": args.gradient_clip_norm,
        "validation_subjects": args.validation_subjects,
        "seed": args.seed,
        "seed_strategy": args.seed_strategy,
        "device": args.device,
        "num_workers": args.num_workers,
        "save_outputs": not args.no_save,
        "run_name": args.run_name,
        "overwrite": args.overwrite,
    }
    if args.output_dir is not None:
        options["output_dir"] = args.output_dir
    config = TrainingConfig(**options)
    if args.all_subjects:
        _, summary = train_all_loso_folds(config)
        print(json.dumps(summary, indent=2))
    else:
        if args.test_subject is None:
            raise RuntimeError("A test subject was not resolved by the parser")
        result = train_loso_fold(args.test_subject, config)
        print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
