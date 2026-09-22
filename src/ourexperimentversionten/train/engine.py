"""One training epoch and the existing accuracy evaluation."""
from dataclasses import dataclass
from typing import Literal, overload

import torch


def train_epoch(model, loader, optimizer, criterion, device) -> None:
    model.train()
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    labels: list[int]
    predictions: list[int]


@overload
def evaluate(model, loader, device, *, collect_predictions: Literal[False] = False) -> float: ...


@overload
def evaluate(model, loader, device, *, collect_predictions: Literal[True]) -> Evaluation: ...


def evaluate(model, loader, device, *, collect_predictions: bool = False) -> float | Evaluation:
    model.eval()
    correct = 0
    labels: list[int] = []
    predictions: list[int] = []
    sample_count = len(loader.dataset)
    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            if collect_predictions:
                labels.extend(data.y.detach().cpu().tolist())
                predictions.extend(pred.detach().cpu().tolist())
    accuracy = correct / sample_count
    return Evaluation(accuracy, labels, predictions) if collect_predictions else accuracy
