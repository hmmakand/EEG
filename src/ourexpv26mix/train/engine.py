"""One training epoch, the seed ensemble, and the scoring of a fold after every epoch."""
import torch
from torch import nn

from .metrics import classification_metrics


def predict(model, data):
    return model(data.x, data.edge_index, data.batch, data.edge_attr, data.signal, getattr(data, "signal_low", None))


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


@torch.no_grad()
def test_logits(model, batches, device) -> torch.Tensor:
    """Logits [trials, classes] of the test batches. Draws no random numbers, so
    calling it between epochs does not alter training."""
    model.eval()
    return torch.cat([predict(model, data.to(device)) for data in batches]).cpu()


def ensemble_predictions(member_logits: torch.Tensor, center_logits: bool) -> torch.Tensor:
    """Predicted classes [trials] from the logits of the models [models, trials, classes].

    The same decision as `evaluate` on an `Ensemble`: the class probabilities
    are averaged, and `center_logits` subtracts the mean over the trials.
    """
    scores = member_logits[0] if len(member_logits) == 1 else member_logits.softmax(dim=-1).mean(dim=0).log()
    if center_logits:
        scores = scores - scores.mean(dim=0, keepdim=True)
    return scores.argmax(dim=1)


def score_epochs(logits: torch.Tensor, labels, center_logits: bool) -> dict:
    """Score a fold after every epoch. `logits` is [models, epochs, trials, classes].

    The returned metrics are those of the final epoch: the result. Two entries
    are added for comparison with work that reports differently:

    highest_epoch            the metrics of the epoch with the highest test
                             accuracy (the earliest on a tie). The epoch is
                             chosen with the test labels, so this score is
                             optimistically biased.
    test_accuracy_per_epoch  the test accuracy after every epoch
    """
    per_epoch = [classification_metrics(labels, ensemble_predictions(logits[:, epoch], center_logits).tolist())
                 for epoch in range(logits.shape[1])]
    accuracies = [metrics["accuracy"] for metrics in per_epoch]
    best = accuracies.index(max(accuracies))
    return dict(per_epoch[-1], highest_epoch=dict(per_epoch[best], epoch=best + 1),
                test_accuracy_per_epoch=accuracies)
