"""PA-TCNet on the command line: its options, its result file name and its summary table."""
from dataclasses import replace

from ...evaluation.metrics import METRICS, format_metrics
from . import experiment
from .settings import DATASETS, TrainingSettings

SAVES_MODELS = False


def add_arguments(parser) -> None:
    parser.add_argument("--protocol", choices=experiment.PROTOCOLS, default="loso")
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--cohort", choices=experiment.COHORTS, default="version27",
                        help="version27 (default): same subjects, tasks, samples and folds as version 27; "
                             "published/distinct: original paper inputs (Liu 24/23 patients)")
    parser.add_argument("--subjects", type=int, nargs="+", help="Held-out subjects to run (default: all)")
    parser.add_argument("--epochs", type=int, help="Epochs per fold (default: 200, the paper's maximum)")


def run(args) -> dict:
    training = replace(TrainingSettings(), **({"epochs": args.epochs} if args.epochs is not None else {}))
    return experiment.run(args.data_dir, args.dataset, args.cohort, args.subjects, training=training,
                          show_progress=not args.quiet, protocol=args.protocol)


def file_stem(args, result) -> str:
    protocol = "LOSO" if args.protocol == "loso" else "Within"
    return f"PATCNet_{args.dataset}_{args.cohort}_{protocol}_seed{result['training']['seed']}"


def summary_text(result: dict) -> list[str]:
    lines = [f"{result['paper']}, {result['paper_dataset']} ({result['dataset']}), cohort {result['cohort']}, "
             f"{result['protocol']}, seed {result['training']['seed']}, PGTC: {result['pgtc']}", ""]
    for subject, fold in result["results"].items():
        lines.append(f"S{subject:<3} final epoch    {format_metrics(fold['final_epoch'])}")
        lines.append(f"     test-selected  {format_metrics(fold['as_published'])}")
    for name in ("final_epoch", "as_published"):
        lines.append(f"\nMean, {name.replace('_', ' ')}: " + format_metrics({k: result['summary'][name][k] for k in METRICS}))
    lines += ["", result["evaluation"]["final_epoch"], result["evaluation"]["as_published"]]
    return lines
