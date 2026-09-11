"""Command-line entry point for the within-subject classification diagnostic.

See ``WITHIN_SUBJECT_PLAN.md`` for why this is a separate diagnostic tool
(train/evaluate on one subject's own trials only, no cross-subject
generalization) rather than a mode of ``train.py``/``search_train.py``. No
``--patience``-style flag exists here by design: each fold trains for a
fixed ``--epochs`` budget and checkpoints on best training loss, since there
is no held-out validation split to early-stop against.
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
from src.ourexperimentversionfour.training.config import WithinSubjectConfig
from src.ourexperimentversionfour.training.within_subject import train_all_within_subject


def _parse_subjects(raw: str) -> list[int]:
    try:
        return [int(item) for item in raw.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--subjects must be a comma-separated list of integers, got {raw!r}"
        ) from None


def build_parser() -> argparse.ArgumentParser:
    """Build the within-subject diagnostic command-line interface."""

    defaults = WithinSubjectConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combination",
        choices=sorted(COMBINATIONS),
        default=defaults.combination,
        help="which node/edge/band feature combination to train on",
    )
    subject_group = parser.add_mutually_exclusive_group(required=True)
    subject_group.add_argument("--subject", type=int, help="run one subject")
    subject_group.add_argument(
        "--all-subjects", action="store_true", help="run every subject in the dataset"
    )
    subject_group.add_argument(
        "--subjects",
        type=_parse_subjects,
        help="comma-separated subject IDs for a fast few-subject pilot, e.g. 1,2,3",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=defaults.folds,
        help="within-subject stratified k-fold count",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=defaults.repeats,
        help="re-run the k-fold split this many times with a different seed and aggregate",
    )
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
    parser.add_argument("--seed", type=int, default=defaults.seed)
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
    """Parse options, run the requested subject scope, and print JSON to stdout."""

    args = build_parser().parse_args(argv)
    options = {
        "combination": args.combination,
        "folds": args.folds,
        "repeats": args.repeats,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "momentum": args.momentum,
        "gradient_clip_norm": args.gradient_clip_norm,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "save_outputs": not args.no_save,
        "run_name": args.run_name,
        "overwrite": args.overwrite,
    }
    if args.output_dir is not None:
        options["output_dir"] = args.output_dir
    config = WithinSubjectConfig(**options)

    if args.all_subjects:
        subject_ids = None
    elif args.subjects is not None:
        subject_ids = args.subjects
    else:
        if args.subject is None:
            raise RuntimeError("A subject was not resolved by the parser")
        subject_ids = [args.subject]

    results, summary = train_all_within_subject(config, subject_ids=subject_ids)
    if args.all_subjects:
        print(json.dumps(summary, indent=2))
    else:
        print(
            json.dumps(
                {
                    "results": [result.as_dict() for result in results],
                    "summary": summary,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
