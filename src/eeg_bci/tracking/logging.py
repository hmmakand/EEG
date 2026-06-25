from __future__ import annotations

import logging
from pathlib import Path

import mne


def configure_logging(log_dir: Path, *, level: int = logging.INFO) -> None:
    """Configure root logging for a Hydra run and write a project log file."""

    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_path = log_dir / "run.log"
    if not _has_file_handler(root, file_path):
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name in (
        "braindecode",
        "braindecode.eegneuralnet.EEGClassifier",
        "mne",
        "skorch",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    mne.set_log_level("WARNING")


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    target = path.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == target:
                return True
    return False

