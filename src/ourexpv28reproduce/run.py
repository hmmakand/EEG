"""Run a paper's experiment and save the result.

    python -m src.ourexpv28reproduce.run --paper patcnet --dataset jia2019 --data-dir /path --output-dir outputs
    python -m src.ourexpv28reproduce.run --paper eegnet --dataset liu2024 --protocol loso --data-dir /path --output-dir outputs
    python -m src.ourexpv28reproduce.run --paper mshallowconvnet --dataset bci2a --protocol within --data-dir /path ...

The result goes to <output-dir>/<paper>/<dataset>/ as JSON and as a readable table.
Each paper has its own options: python -m src.ourexpv28reproduce.run --paper eegnet --help
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .papers.conformer import command as conformer
from .papers.dbconformer import command as dbconformer
from .papers.eegnet import command as eegnet
from .papers.mscformer import command as mscformer
from .papers.mshallowconvnet import command as mshallowconvnet
from .papers.patcnet import command as patcnet
from .papers.slimseiz import command as slimseiz
from .papers.uadaan import command as uadaan

PAPERS = dict(patcnet=patcnet, eegnet=eegnet, mshallowconvnet=mshallowconvnet, conformer=conformer,
              mscformer=mscformer, dbconformer=dbconformer, slimseiz=slimseiz,
              uadaan=uadaan)


def cost_lines(result: dict) -> list[str]:
    size, cost = result["complexity"], result["cost"]
    memory = cost["peak_gpu_memory_bytes"]
    return ["", f"Model ({result['device_name']}): {size['parameters']:,} parameters, "
                f"{size['flops_per_trial'] / 1e6:.2f} MFLOPs per trial, file {size['model_file_bytes'] / 1024:.0f} KiB",
            f"Inference: {size['latency_ms_per_trial']:.2f} ms per trial alone, "
            f"{size['latency_ms_per_trial_in_batch']:.3f} ms per trial in a batch of {size['latency_batch_size']}",
            f"Training: {cost['training_seconds_total'] / 60:.1f} min in all, "
            f"{cost['training_seconds_per_fold_mean']:.1f} s per fold of {cost['epochs_per_fold']} epochs"
            + (f", peak GPU memory {memory / 2 ** 20:.0f} MiB" if memory else "")]


def main(argv=None):
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--paper", choices=tuple(PAPERS), default="patcnet")
    paper = PAPERS[first.parse_known_args(argv)[0].paper]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", choices=tuple(PAPERS), default="patcnet")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--save-models", action="store_true",
                        help="Save the trained models next to the result (all papers but patcnet)")
    parser.add_argument("--quiet", action="store_true", help="Hide the progress bars")
    paper.add_arguments(parser)
    args = parser.parse_args(argv)

    folder = args.output_dir / args.paper / args.dataset
    args.models_dir = None
    if args.save_models and not paper.SAVES_MODELS:
        parser.error(f"--save-models is not available for {args.paper}")
    try:
        if args.save_models:
            args.models_dir = folder / "models" / f"pending_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        result = paper.run(args)
    except ValueError as error:
        parser.error(str(error))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{paper.file_stem(args, result)}.json"
    if path.exists():
        path = path.with_name(f"{path.stem}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    path.with_name(path.stem + ".summary.txt").write_text("\n".join(paper.summary_text(result) + cost_lines(result))
                                                         + "\n")
    print(f"Result saved to {path}")
    if args.models_dir is not None and args.models_dir.exists():
        target = folder / "models" / path.stem
        args.models_dir.rename(target)
        print(f"Models saved to {target}")


if __name__ == "__main__":
    main()
