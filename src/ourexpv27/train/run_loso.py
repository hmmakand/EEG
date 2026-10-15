"""Run the leave-one-subject-out experiment and save the result as JSON.

    python -m src.ourexpv27.train.run_loso --data-dir /path/to/MNE-liu2024-data --output-dir outputs
    python -m src.ourexpv27.train.run_loso --dataset bci2a --data-dir /path/to/A0xT.mat/files --output-dir outputs

The result goes to <output-dir>/<dataset>/.
"""
import argparse
from dataclasses import replace
from pathlib import Path

from ..datascripts import DATASETS, dataset_config, subject_file
from ..model.config import PATHS
from .config import TrainConfig
from .loso import run_loso
from .output import models_folder, result_path, write_result


def add_common_arguments(parser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, default="liu2024",
                        help="liu2024: imagery against rest (default); bci2a and jia2019: left against right hand")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="liu2024: the MNE-liu2024-data folder; bci2a: the folder with A01T.mat ... A09T.mat; "
                             "jia2019: the folder with exp1-S1-left.mat ...")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Results go to <output-dir>/<dataset>/ (default: ./outputs)")
    parser.add_argument("--output", type=Path, help="Exact result file, instead of the name chosen in --output-dir")
    parser.add_argument("--save-models", action="store_true",
                        help="Save every trained model after its final epoch, next to the result file")
    parser.add_argument("--subjects", type=int, nargs="+", help="Subjects to run (default: all of the dataset)")
    parser.add_argument("--seed", type=int, help="Seed of the first model of each fold (default: 12345)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--quiet", action="store_true", help="Hide the per-model progress bars")
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=list(PATHS),
                        help="Ablation: the paths of the model whose logits are added (default: all three, "
                             "the method)")


def variant_name(paths, target_calibration: bool = True) -> str:
    """The part of the result file name that marks an ablation; empty for the method itself."""
    parts = []
    if set(paths) != set(PATHS):
        parts.append("Paths-" + "-".join(name for name in PATHS if name in paths))
    if not target_calibration:
        parts.append("NoCalibration")
    return "_".join(parts)


def check_recordings(parser, args, data_config, subjects) -> None:
    missing = [subject_file(args.data_dir, s, data_config).name for s in subjects
               if not subject_file(args.data_dir, s, data_config).is_file()]
    if missing:
        parser.error(f"Missing recordings in {args.data_dir}: {', '.join(missing)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--no-target-calibration", action="store_true",
                        help="Ablation: train without target calibration on the held-out subject")
    args = parser.parse_args(argv)

    data_config = dataset_config(args.dataset)
    overrides = {name: getattr(args, name) for name in ("seed", "device") if getattr(args, name) is not None}
    if args.no_target_calibration:
        overrides["target_calibration"] = False
    config = replace(TrainConfig(), **overrides, show_progress=not args.quiet)
    check_recordings(parser, args, data_config, data_config.subjects)
    output = args.output or result_path(args.output_dir, args.dataset, "LOSO", config.seed,
                                        variant_name(args.paths, not args.no_target_calibration))
    try:
        result = run_loso(args.data_dir, data_config, config, held_out_subjects=args.subjects,
                          models_dir=models_folder(output) if args.save_models else None, paths=args.paths)
    except ValueError as error:
        parser.error(str(error))
    write_result(result, output)


if __name__ == "__main__":
    main()
