"""Shared command-line interface for the four thin CFSPMNet scripts."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from omegaconf import OmegaConf

from eeg_bci.cfspmnet.experiment import (
    CFSPMExperimentConfig,
    current_result_rows,
    run_cfspmnet_experiment,
    scientific_config_fingerprint,
    summarize_current_results,
    validate_experiment_config,
)
from eeg_bci.tracking.logging import configure_logging

SMOKE_HELD_OUT_SUBJECTS = [1]


def run_cli(
    base_config: CFSPMExperimentConfig,
    argv: Sequence[str] | None = None,
) -> int:
    args = _parser(base_config).parse_args(argv)
    config = replace(
        base_config,
        held_out_subject_ids=args.subjects,
        stage_i_epochs=args.stage_i_epochs,
        stage_ii_epochs=args.stage_ii_epochs,
        device=args.device,
        resume=not args.no_resume,
        apply_ica=base_config.apply_ica and not args.skip_ica,
        download_if_missing=not args.no_download,
        figshare_cache_dir=args.figshare_cache_dir,
        output_root=args.output_root,
    )
    validate_experiment_config(config)
    fingerprint = scientific_config_fingerprint(config)

    if args.print_config:
        print(
            OmegaConf.to_yaml(
                OmegaConf.create(
                    {
                        "configuration": asdict(config),
                        "config_fingerprint": fingerprint,
                    }
                ),
                resolve=True,
            )
        )
        return 0

    repo_root = _find_repo_root(Path(__file__))
    output_root = (
        repo_root / "outputs"
        if config.output_root is None
        else (
            config.output_root
            if config.output_root.is_absolute()
            else repo_root / config.output_root
        )
    )
    configure_logging(output_root / "logs" / config.experiment_name)
    _add_console_logging()

    print(
        f"{config.experiment_name}: protocol={config.protocol}, "
        f"canonicalization={config.canonicalization}, "
        f"preprocessing={config.preprocessing}, cohort={len(config.subject_ids)}, "
        f"held_out={config.effective_held_out_subject_ids}, "
        f"epochs=stage-I:{config.stage_i_epochs}+stage-II:{config.stage_ii_epochs}"
        f"=total:{config.total_epochs}, cfg={fingerprint}"
    )
    if (
        config.stage_i_epochs != base_config.stage_i_epochs
        or config.stage_ii_epochs != base_config.stage_ii_epochs
    ):
        print(
            "Epoch override active: "
            f"locked preset={base_config.stage_i_epochs}+{base_config.stage_ii_epochs}; "
            f"this run={config.stage_i_epochs}+{config.stage_ii_epochs}."
        )
        if config.total_epochs < base_config.total_epochs:
            print(
                "Reduced epoch budgets are suitable for pipeline diagnostics, "
                "not for estimating the locked preset's final accuracy."
            )
    new_rows = run_cfspmnet_experiment(config)
    rows = current_result_rows(config)
    summary = summarize_current_results(config)
    print(f"Completed folds for cfg {fingerprint}: {len(rows)}")
    for key, value in sorted(summary.items()):
        if key.endswith("_mean") or key.endswith("_std"):
            print(f"  {key:<32} {value:.4f}")

    failed = [
        int(row["held_out_subject"])
        for row in new_rows
        if row.get("status") != "success"
    ]
    if failed:
        print(f"Failed held-out folds: {failed}", file=sys.stderr)
        return 1
    return 0


def _parser(base_config: CFSPMExperimentConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Run {base_config.experiment_name} on the fixed 50-subject Liu2024 "
            "cohort. --subjects selects folds to execute, not the source cohort."
        )
    )
    parser.add_argument(
        "--subjects",
        type=_parse_subjects,
        default=None,
        help=(
            "'all', 'smoke' (held-out subject 1), or comma-separated held-out "
            "subject IDs. The training cohort always remains subjects 1..50."
        ),
    )
    parser.add_argument(
        "--stage-i-epochs",
        type=int,
        default=base_config.stage_i_epochs,
    )
    parser.add_argument(
        "--stage-ii-epochs",
        type=int,
        default=base_config.stage_ii_epochs,
    )
    parser.add_argument("--device", default=base_config.device)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-ica",
        action="store_true",
        help="Diagnostic-only override for the paper-aligned raw profile.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading a missing Figshare raw archive.",
    )
    parser.add_argument("--figshare-cache-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved configuration and exit without loading data.",
    )
    return parser


def _parse_subjects(value: str) -> list[int]:
    if value == "all":
        return list(range(1, 51))
    if value == "smoke":
        return list(SMOKE_HELD_OUT_SUBJECTS)
    parsed = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("At least one held-out subject is required.")
    invalid = [subject_id for subject_id in parsed if subject_id not in range(1, 51)]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Liu2024 subject IDs must be in 1..50; got {invalid}."
        )
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("Held-out subject IDs must be unique.")
    return parsed


def _add_console_logging() -> None:
    root = logging.getLogger()
    has_console = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root.handlers
    )
    if has_console:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")
