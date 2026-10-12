"""One training epoch, the seed ensemble, and evaluation of a held-out subject."""
import torch
from torch import nn


def predict(model, data):
    return model(data.x, data.edge_index, data.batch)


def train_epoch(model, loader, optimizer, criterion, device) -> None:
    model.train()
    for data in loader:
        data = data.to(device)
        loss = criterion(predict(model, data), data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class Ensemble(nn.Module):
    """Average the class probabilities of independently trained models."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *inputs):
        probabilities = torch.stack([model(*inputs).softmax(dim=-1) for model in self.models])
        return probabilities.mean(dim=0).log()


@torch.no_grad()
def evaluate(model, loader, device, center_logits: bool):
    """Return (labels, predictions) for every trial of the loader.

    `center_logits` subtracts the mean logit of the evaluated trials before
    the decision. It removes a subject-wide bias toward one class without
    using labels, and assumes the loader holds one subject.
    """
    model.eval()
    outputs, labels = [], []
    for data in loader:
        data = data.to(device)
        outputs.append(predict(model, data))
        labels.append(data.y)
    if not outputs:
        raise ValueError("Cannot evaluate an empty dataset")
    logits, labels = torch.cat(outputs), torch.cat(labels)
    if center_logits:
        logits = logits - logits.mean(dim=0, keepdim=True)
    return labels.cpu().tolist(), logits.argmax(dim=1).cpu().tolist()
