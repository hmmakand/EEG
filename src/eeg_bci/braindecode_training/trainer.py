from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from eeg_bci.data.splitting import split_train_valid
from eeg_bci.braindecode_training.checkpointing import save_classifier_module
from eeg_bci.braindecode_training.classifier import build_eeg_classifier
from eeg_bci.braindecode_training.evaluation import latest_history_value, score_classifier


def train_model(
    model: nn.Module,
    train_set: Dataset,
    test_set: Dataset,
    *,
    n_outputs: int,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    output_dir: Path,
    checkpoint_name: str,
    validation_enabled: bool,
    validation_size: float,
    validation_shuffle: bool,
    seed: int,
) -> dict[str, float]:
    valid_set: Dataset | None = None
    if validation_enabled:
        train_set, valid_set = split_train_valid(
            train_set,
            valid_size=validation_size,
            shuffle=validation_shuffle,
            seed=seed,
        )

    classifier = build_eeg_classifier(
        model,
        n_outputs=n_outputs,
        device=device,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_workers=num_workers,
        valid_set=valid_set,
    )

    classifier.fit(train_set, y=None)
    test_acc = score_classifier(classifier, test_set)
    train_loss = latest_history_value(classifier, "train_loss")
    save_classifier_module(classifier, output_dir, checkpoint_name)

    metrics = {
        "train_loss": train_loss,
        "test_acc": test_acc,
    }
    if valid_set is not None:
        metrics["valid_acc"] = latest_history_value(classifier, "valid_accuracy")

    print(f"test_acc={test_acc:.4f}")
    return metrics
