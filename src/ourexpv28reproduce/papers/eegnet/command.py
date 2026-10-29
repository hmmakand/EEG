"""EEGNet on the command line: its options, its result file name and its summary table."""
from dataclasses import replace

from ...datascripts.version27 import DATASETS
from ...evaluation.summary import scores_summary
from . import experiment

SAVES_MODELS = True


def add_arguments(parser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--protocol", choices=experiment.PROTOCOLS, default="loso",
                        help="loso: leave one subject out (default); within: version 27's 10 folds per subject")
    parser.add_argument("--subjects", type=int, nargs="+",
                        help="Subjects to run: held-out subjects (loso) or subjects (within). Default: all")
    parser.add_argument("--epochs", type=int, help="Epochs per fold (default: 500, the paper's)")
    parser.add_argument("--seed", type=int, help="Seed (default: 12345)")


def run(args) -> dict:
    training = experiment.TrainingSettings()
    overrides = {name: getattr(args, name) for name in ("epochs", "seed") if getattr(args, name) is not None}
    return experiment.run(args.data_dir, args.dataset, args.protocol, args.subjects,
                          training=replace(training, **overrides), show_progress=not args.quiet,
                          models_dir=args.models_dir)


def file_stem(args, result) -> str:
    protocol = "LOSO" if args.protocol == "loso" else "Within"
    return f"EEGNet_{args.dataset}_{protocol}_seed{result['training']['seed']}"


summary_text = scores_summary
