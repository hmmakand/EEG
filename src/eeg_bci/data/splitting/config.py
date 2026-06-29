"""Small helpers for reading split configuration values."""

from __future__ import annotations

from omegaconf import DictConfig

from eeg_bci.data.splitting.strategies import SESSION_TRAIN_TEST


def section(split_cfg: DictConfig, name: str) -> DictConfig:
    value = split_cfg.get(name, None)
    return value if isinstance(value, DictConfig) else split_cfg


def configured_strategy(split_cfg: DictConfig | None) -> str:
    if split_cfg is None:
        return "random"
    return str(split_cfg.get("split_strategy", split_cfg.get("strategy", "random")))


def session_split_strategy(split_cfg: DictConfig) -> str:
    strategy = configured_strategy(split_cfg)
    if strategy == "description":
        return SESSION_TRAIN_TEST
    return strategy


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
