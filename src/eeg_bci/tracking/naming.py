from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

from omegaconf import DictConfig, OmegaConf

from eeg_bci.data.splitting import resolved_split_label


def slugify(value: object) -> str:
    """Return a filesystem-friendly identifier."""

    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def dataset_label(dataset_cfg: DictConfig) -> str:
    if "label" in dataset_cfg:
        return slugify(dataset_cfg.label)
    if "dataset_name" in dataset_cfg:
        return slugify(dataset_cfg.dataset_name)
    if "name" in dataset_cfg:
        return slugify(dataset_cfg.name)
    return "dataset"


def model_label(model_cfg: DictConfig) -> str:
    
    if "label" in model_cfg:
        return slugify(model_cfg.label)
    return slugify(model_cfg.name if "name" in model_cfg else "model")


def subject_label(subject_id: int | str) -> str:
    if isinstance(subject_id, int) or str(subject_id).isdigit():
        return f"s{int(subject_id):02d}"
    return slugify(subject_id)


def held_out_label(subject_id: int | str) -> str:
    return f"heldout-{subject_label(subject_id)}"


def subject_scope(subject_ids: Any) -> str:
    values = OmegaConf.to_container(subject_ids, resolve=True)
    if values is None:
        return "all_subjects"
    if isinstance(values, int):
        return subject_label(values)
    if isinstance(values, list):
        if len(values) == 1:
            return subject_label(values[0])
        numeric = [int(value) for value in values if str(value).isdigit()]
        if len(numeric) == len(values):
            return f"{subject_label(min(numeric))}-{subject_label(max(numeric))}"
        return "-".join(subject_label(value) for value in values)
    return slugify(values)


def experiment_label(cfg: DictConfig) -> str:
    if "experiment_name" in cfg:
        return slugify(cfg.experiment_name)
    if "dataset" in cfg and "split" in cfg.dataset:
        return slugify(resolved_split_label(cfg.dataset.split))
    return "run"


def class_names_from_mapping(dataset_cfg: DictConfig) -> list[str] | None:
    """Return class names ordered by their integer label index.

    Looks for ``mapping`` in the dataset config, e.g.
    ``left_hand: 0, right_hand: 1, ...``. Returns ``None`` if no mapping exists.
    """
    if "mapping" not in dataset_cfg:
        return None
    mapping = OmegaConf.to_container(dataset_cfg.mapping, resolve=True)
    if not isinstance(mapping, dict):
        return None
    sorted_items = sorted(mapping.items(), key=lambda item: int(item[1]))
    return [str(name) for name, _ in sorted_items]


def run_id(scope: str, seed: int, *, created_at: str | None = None) -> str:
    return f"{slugify(scope)}__seed{seed}__{created_at or timestamp()}"


def tensorboard_dir(
    cfg: DictConfig,
    run_name: str,
    scope: str | None = None,
    *,
    base_dir: str = "outputs/tensorboard",
) -> Path:
    path = (
        Path(base_dir)
        / dataset_label(cfg.dataset)
        / experiment_label(cfg)
        / str(cfg.dataset.split.method)
        / model_label(cfg.model)
        / run_name
    )
    if scope is not None:
        path = path / str(scope)
    return path

