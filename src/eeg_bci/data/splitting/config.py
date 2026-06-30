"""Small helpers for reading split configuration values."""

from __future__ import annotations

from omegaconf import DictConfig

from eeg_bci.data.splitting.strategies import SOURCE_RANDOM, TRAIN_TEST, split_label


def section(split_cfg: DictConfig, name: str) -> DictConfig:
    value = split_cfg.get(name, None)
    return value if isinstance(value, DictConfig) else split_cfg


def resolved_source_and_method(split_cfg: DictConfig | None) -> tuple[str, str]:
    """Resolve a dataset's split config into a (source, method) pair.

    A dataset's ``split`` config must set both ``source`` and ``method``
    explicitly (see ``splitting/strategies.py``). A missing ``split`` block
    entirely defaults to a plain random train/test split.
    """

    if split_cfg is None:
        return SOURCE_RANDOM, TRAIN_TEST

    source = split_cfg.get("source", None)
    method = split_cfg.get("method", None)
    if source is None or method is None:
        raise ValueError(
            "dataset.split must set both 'source' and 'method', e.g. "
            "{source: session, method: train_valid_test}."
        )
    return str(source), str(method)


def resolved_split_label(split_cfg: DictConfig | None) -> str:
    """Human-readable (source, method) label for logging/display."""

    return split_label(*resolved_source_and_method(split_cfg))


def validation_size(split_cfg: DictConfig) -> float:
    validation_cfg = section(split_cfg, "validation")
    return float(
        validation_cfg.get(
            "valid_size",
            split_cfg.get("valid_size", split_cfg.get("validation_size", 0.2)),
        )
    )


def validation_shuffle(split_cfg: DictConfig) -> bool:
    validation_cfg = section(split_cfg, "validation")
    return bool(
        validation_cfg.get(
            "shuffle",
            split_cfg.get("shuffle", split_cfg.get("validation_shuffle", False)),
        )
    )


def split_lengths(total_len: int, *, holdout_size: float) -> tuple[int, int]:
    if total_len < 2:
        raise ValueError("At least two samples are required to create a split.")
    if not 0.0 < holdout_size < 1.0:
        raise ValueError("Holdout size must be between 0 and 1.")

    train_len = int(round(total_len * (1.0 - holdout_size)))
    train_len = min(max(train_len, 1), total_len - 1)
    holdout_len = total_len - train_len
    return train_len, holdout_len
