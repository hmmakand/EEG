"""One training epoch and the existing accuracy evaluation."""
from dataclasses import dataclass
from typing import Literal, overload

import torch


def predict(model, data, **extra):
    # Only the custom model consumes edge features; GATv2 takes the topology.
    if getattr(model, "uses_signal", False) is True:
        extra["signal"] = data.signal
    if getattr(model, "uses_edge_attr", False) is True:
        edge_attr = data.get("edge_attr")
        return model(data.x, data.edge_index, data.batch,
                     edge_attr if edge_attr is not None else data.edge_weight, **extra)
    if extra:
        return model(data.x, data.edge_index, data.batch, **extra)
    return model(data.x, data.edge_index, data.batch)


def train_epoch(model, loader, optimizer, criterion, device, invariance=None, progress: float = 0.0) -> None:
    """`invariance` (train/invariance.py) adds its loss on the model's features;
    `progress` is the fraction of training completed, in [0, 1]."""
    model.train()
    if invariance is not None:
        invariance.train()
    for data in loader:
        data = data.to(device)
        if invariance is None:
            out = predict(model, data)
            loss = criterion(out, data.y)
        else:
            out, features = predict(model, data, return_features=True)
            loss = criterion(out, data.y) + invariance(features, data.y, data.subject_id, progress)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    labels: list[int]
    predictions: list[int]


@overload
def evaluate(model, loader, device, *, collect_predictions: Literal[False] = False,
             center_logits: bool = False) -> float: ...


@overload
def evaluate(model, loader, device, *, collect_predictions: Literal[True],
             center_logits: bool = False) -> Evaluation: ...


def evaluate(model, loader, device, *, collect_predictions: bool = False,
             center_logits: bool = False) -> float | Evaluation:
    """Accuracy over the loader.

    `center_logits` subtracts the mean logit of the evaluated set from every
    sample before the decision. It removes a subject-wide bias toward one
    class without using labels, and assumes the set comes from one subject.
    """
    model.eval()
    sample_count = len(loader.dataset)
    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    outputs, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs.append(predict(model, data))
            targets.append(data.y)
    out, target = torch.cat(outputs), torch.cat(targets)
    if center_logits:
        out = out - out.mean(dim=0, keepdim=True)
    pred = out.argmax(dim=1)
    accuracy = int((pred == target).sum()) / sample_count
    if collect_predictions:
        return Evaluation(accuracy, target.detach().cpu().tolist(), pred.detach().cpu().tolist())
    return accuracy
