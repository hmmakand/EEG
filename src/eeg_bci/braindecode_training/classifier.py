from __future__ import annotations

import logging

import torch
from braindecode import EEGClassifier
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from torch import nn
from torch.utils.data import Dataset


def build_eeg_classifier(
    model: nn.Module,
    *,
    n_outputs: int,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_workers: int,
    valid_set: Dataset | None = None,
) -> EEGClassifier:
    logging.getLogger("braindecode.eegneuralnet.EEGClassifier").setLevel(logging.WARNING)

    callbacks = [
        "accuracy",
        (
            "lr_scheduler",
            LRScheduler("CosineAnnealingLR", T_max=max(max_epochs - 1, 1)),
        ),
    ]
    train_split = predefined_split(valid_set) if valid_set is not None else None
    return EEGClassifier(
        model,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        train_split=train_split,
        optimizer__lr=learning_rate,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=callbacks,
        device=str(device),
        classes=list(range(n_outputs)),
        max_epochs=max_epochs,
        iterator_train__num_workers=num_workers,
        iterator_train__drop_last=False,
        iterator_valid__num_workers=num_workers,
        iterator_valid__drop_last=False,
    )
