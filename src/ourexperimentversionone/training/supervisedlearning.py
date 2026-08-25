"""Simple full strict-LOSO EEGNet training for Liu2024."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    repository_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository_root))

from src.ourexperimentversionone.data import (  # noqa: E402
    DataLoaderConfig,
    create_loso_dataloaders,
)
from src.ourexperimentversionone.model.EEGNet import EEGNet  # noqa: E402


def set_seed(seed: int) -> None:
    """Seed the random generators used during training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or evaluation epoch and return loss and accuracy."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = loss_function(logits, y)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_loso(
    *,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 42,
    device: str | None = None,
) -> list[dict[str, float | int]]:
    """Run strict LOSO training for all 50 Liu2024 subjects."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    loader_config = DataLoaderConfig(
        batch_size=batch_size,
        pin_memory=selected_device.type == "cuda",
        seed=seed,
    )
    loss_function = nn.CrossEntropyLoss()
    print(f"Device: {selected_device}")
    print(f"LOSO folds: 50")
    print(f"Epochs per fold: {epochs}")
    results: list[dict[str, float | int]] = []

    for test_subject_id in range(1, 51):
        # Every fold starts from a fresh, reproducibly initialized model.
        set_seed(seed)
        loaders = create_loso_dataloaders(
            test_subject_id=test_subject_id,
            config=loader_config,
        )
        model = EEGNet().to(selected_device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        print(
            f"\nFold {test_subject_id:02d}/50 | "
            f"train subjects: {len(loaders.split.train_subject_ids)} | "
            f"test subject: {test_subject_id}"
        )

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = run_epoch(
                model, loaders.train, loss_function, selected_device, optimizer
            )
            print(
                f"  Epoch {epoch:03d} | "
                f"train loss {train_loss:.4f}, accuracy {train_accuracy:.3f}"
            )

        test_loss, test_accuracy = run_epoch(
            model, loaders.test, loss_function, selected_device
        )
        results.append(
            {
                "test_subject_id": test_subject_id,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        print(f"  Test loss: {test_loss:.4f}")
        print(f"  Test accuracy: {test_accuracy:.3f}")

    accuracies = np.asarray(
        [result["test_accuracy"] for result in results], dtype=np.float64
    )
    print("\nFull strict LOSO complete")
    print(f"Mean accuracy: {accuracies.mean():.3f}")
    print(f"Standard deviation: {accuracies.std():.3f}")
    print(f"Median accuracy: {np.median(accuracies):.3f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    train_loso(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
