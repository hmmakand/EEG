"""Reusable training and evaluation passes for graph classifiers."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .metrics import ClassificationMetrics, calculate_metrics


DenseBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
"""One (x, adj, labels) batch as produced by the dense EEGGCN1 DataLoader."""


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve an explicit device or select CUDA when available."""

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def _labels(labels: torch.Tensor) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError("Every batch must contain tensor labels")
    if labels.ndim != 1:
        labels = labels.reshape(-1)
    if labels.dtype != torch.long:
        labels = labels.to(dtype=torch.long)
    return labels


def _run_loader(
    model: nn.Module,
    loader: Iterable[DenseBatch],
    loss_function: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float | None,
) -> ClassificationMetrics:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for x, adjacency, raw_labels in loader:
            x = x.to(device, non_blocking=True)
            adjacency = adjacency.to(device, non_blocking=True)
            labels = _labels(raw_labels.to(device, non_blocking=True))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            log_probabilities = model(x, adjacency)
            if log_probabilities.ndim != 2 or log_probabilities.shape[0] != labels.numel():
                raise ValueError(
                    "The model must return one class-probability row per graph label"
                )
            loss = loss_function(log_probabilities, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Encountered a non-finite batch loss")
            if optimizer is not None:
                loss.backward()
                if gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            total_loss += float(loss.detach().item()) * labels.numel()
            all_labels.append(labels.detach().cpu())
            all_predictions.append(log_probabilities.detach().argmax(dim=1).cpu())
            # The model outputs log-probabilities (log-softmax), so recover
            # the positive-class probability via exp, not another softmax.
            all_scores.append(
                torch.exp(log_probabilities.detach())[:, 1].cpu()
            )

    if not all_labels:
        raise ValueError("Cannot run an epoch with an empty DataLoader")
    labels_array = torch.cat(all_labels).numpy()
    predictions_array = torch.cat(all_predictions).numpy()
    scores_array = torch.cat(all_scores).numpy()
    return calculate_metrics(
        total_loss=total_loss,
        labels=labels_array,
        predictions=predictions_array,
        scores=scores_array,
    )


def train_epoch(
    model: nn.Module,
    loader: Iterable[DenseBatch],
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    gradient_clip_norm: float | None = 1.0,
) -> ClassificationMetrics:
    """Train for one complete loader pass and return aggregate metrics."""

    return _run_loader(
        model,
        loader,
        loss_function,
        device,
        optimizer=optimizer,
        gradient_clip_norm=gradient_clip_norm,
    )


def evaluate(
    model: nn.Module,
    loader: Iterable[DenseBatch],
    loss_function: nn.Module,
    device: torch.device,
) -> ClassificationMetrics:
    """Evaluate without gradients and return aggregate metrics."""

    return _run_loader(
        model,
        loader,
        loss_function,
        device,
        optimizer=None,
        gradient_clip_norm=None,
    )
