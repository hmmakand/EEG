"""Run a paper's experiment and save the result.

    python -m src.ourexpv28reproduce.run --paper patcnet --dataset jia2019 --data-dir /path --output-dir outputs

The result goes to <output-dir>/<paper>/<dataset>/ as JSON and as a readable table.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from .evaluation.metrics import METRICS, format_metrics
from .papers.patcnet import experiment as patcnet
from .papers.patcnet.settings import DATASETS, TrainingSettings

PAPERS = dict(patcnet=patcnet)


def summary_text(result: dict) -> str:
    lines = [f"{result['paper']}, {result['paper_dataset']} ({result['dataset']}), cohort {result['cohort']}, "
             f"leave one subject out, seed {result['training']['seed']}, PGTC: {result['pgtc']}", ""]
    for subject, fold in result["results"].items():
        lines.append(f"S{subject:<3} final epoch    {format_metrics(fold['final_epoch'])}")
        lines.append(f"     as published   {format_metrics(fold['as_published'])}   (epoch {fold['as_published']['epoch']})")
    for name in ("final_epoch", "as_published"):
        lines.append(f"\nMean, {name.replace('_', ' ')}: " + format_metrics({k: result['summary'][name][k] for k in METRICS}))
    lines += ["", result["evaluation"]["final_epoch"], result["evaluation"]["as_published"]]
    size, cost = result["complexity"], result["cost"]
    memory = cost["peak_gpu_memory_bytes"]
    lines += ["", f"Model ({result['device_name']}): {size['parameters']:,} parameters, "
                  f"{size['flops_per_trial'] / 1e6:.2f} MFLOPs per trial, file {size['model_file_bytes'] / 1024:.0f} KiB",
              f"Inference: {size['latency_ms_per_trial']:.2f} ms per trial alone, "
              f"{size['latency_ms_per_trial_in_batch']:.3f} ms per trial in a batch of {size['latency_batch_size']}",
              f"Training: {cost['training_seconds_total'] / 60:.1f} min in all, "
              f"{cost['training_seconds_per_fold_mean']:.1f} s per fold of {cost['epochs_per_fold']} epochs"
              + (f", peak GPU memory {memory / 2 ** 20:.0f} MiB" if memory else "")]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", choices=tuple(PAPERS), default="patcnet")
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cohort", choices=patcnet.COHORTS, default="published",
                        help="liu2024: the paper's 24 subjects (published) or 23 without the duplicated recording "
                             "(distinct); jia2019 always has 15")
    parser.add_argument("--subjects", type=int, nargs="+", help="Held-out subjects to run (default: all)")
    parser.add_argument("--epochs", type=int, help="Epochs per fold (default: 200, the paper's maximum)")
    parser.add_argument("--quiet", action="store_true", help="Hide the progress bars")
    args = parser.parse_args(argv)
    training = replace(TrainingSettings(), **({"epochs": args.epochs} if args.epochs else {}))
    try:
        result = PAPERS[args.paper].run(args.data_dir, args.dataset, args.cohort, args.subjects, training=training,
                                        show_progress=not args.quiet)
    except ValueError as error:
        parser.error(str(error))
    folder = args.output_dir / args.paper / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"PATCNet_{args.dataset}_{args.cohort}_LOSO_seed{training.seed}.json"
    if path.exists():
        path = path.with_name(f"{path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    path.with_name(path.stem + ".summary.txt").write_text(summary_text(result))
    print(f"Result saved to {path}")


if __name__ == "__main__":
    main()
