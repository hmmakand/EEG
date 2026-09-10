"""Command-line entry point for LOSO training with inner-CV hyperparameter search.

Separate from ``train.py`` because the cost profile is qualitatively
different: each outer fold now runs ``inner_folds x len(grid)`` extra
training runs before its final retrain (defaults: 10 x 12 = 120 per fold).
See ``training/README.md`` for the expected runtime before running
``--all-subjects`` with the default budget.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.ourexperimentversionfour.data.combinations import COMBINATIONS
from src.ourexperimentversionfour.training.config import TrainingConfig
from src.ourexperimentversionfour.training.loso import (
    train_all_loso_folds_with_search,
    train_loso_fold_with_search,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the inner-CV hyperparameter-search command-line interface."""

    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Note: --learning-rate/--weight-decay set the *base* config, but "
            "the search always overrides exactly those two fields for the "
            "final retrain -- passing them has no effect on the actual "
            "result. Only --batch-size/--gradient-clip-norm/etc. (fields "
            "the default grid doesn't touch) pass through as given."
        ),
    )
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
    patience_group = parser.add_mutually_exclusive_group()
    patience_group.add_argument(
        "--patience", type=int, default=defaults.patience
    )
    patience_group.add_argument(
        "--no-early-stopping",
        action="store_const",
        const=None,
        dest="patience",
        help="disable early stopping on the final retrain and always run "
        "the full --epochs budget",
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
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=10,
        help="number of grouped inner-CV folds to split the 49 development "
        "subjects into for hyperparameter search",
    )
    parser.add_argument(
        "--search-epochs",
        type=int,
        default=50,
        help="max epochs per inner-CV training run (separate from --epochs, "
        "which is the final retrain's budget)",
    )
    search_patience_group = parser.add_mutually_exclusive_group()
    search_patience_group.add_argument(
        "--search-patience",
        type=int,
        default=8,
        help="early-stopping patience for inner-CV runs (separate from "
        "--patience, which applies to the final retrain)",
    )
    search_patience_group.add_argument(
        "--no-search-early-stopping",
        action="store_const",
        const=None,
        dest="search_patience",
        help="disable early stopping on inner-CV runs and always run the "
        "full --search-epochs budget",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse options, run the requested fold scope, and print JSON to stdout."""

    args = build_parser().parse_args(argv)
    options = {
        "combination": args.combination,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
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
    search_kwargs = {
        "inner_folds": args.inner_folds,
        "search_epochs": args.search_epochs,
        "search_patience": args.search_patience,
    }
    if args.all_subjects:
        _, summary = train_all_loso_folds_with_search(config, **search_kwargs)
        print(json.dumps(summary, indent=2))
    else:
        if args.test_subject is None:
            raise RuntimeError("A test subject was not resolved by the parser")
        result, provenance = train_loso_fold_with_search(
            args.test_subject, config, **search_kwargs
        )
        print(
            json.dumps(
                {"result": result.as_dict(), **provenance}, indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
