"""One training epoch, test logits after an epoch, and the two ways of scoring a fold."""
import torch

from .metrics import classification_metrics


def predict(model, data):
    return model(data.x, data.edge_index, data.batch, data.edge_attr, data.signal)


def train_epoch(model, loader, optimizer, criterion, device) -> None:
    model.train()
    for data in loader:
        data = data.to(device)
        loss = criterion(predict(model, data), data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test_logits(model, batches, device) -> torch.Tensor:
    """Logits [trials, classes] of the test batches. Draws no random numbers, so
    calling it between epochs leaves training exactly as it is in version 23."""
    model.eval()
    return torch.cat([predict(model, data.to(device)) for data in batches]).cpu()


def ensemble_predictions(member_logits: torch.Tensor, center_logits: bool) -> torch.Tensor:
    """Predicted classes [trials] from the logits of the models [models, trials, classes].

    The class probabilities of the models are averaged. `center_logits`
    subtracts the mean over the evaluated trials before the decision; it
    removes a subject-wide bias toward one class without using labels.
    """
    scores = member_logits[0] if len(member_logits) == 1 else member_logits.softmax(dim=-1).mean(dim=0).log()
    if center_logits:
        scores = scores - scores.mean(dim=0, keepdim=True)
    return scores.argmax(dim=1)


def score_epochs(logits: torch.Tensor, labels, center_logits: bool) -> dict:
    """Score a fold after every epoch. `logits` is [models, epochs, trials, classes].

    final_epoch    the score after the last epoch. No test label is used to
                   obtain it; this is the result of version 23.
    highest_epoch  the score of the epoch with the highest test accuracy (the
                   earliest one on a tie). The epoch is chosen with the test
                   labels, so this score is optimistically biased.
    """
    per_epoch = [classification_metrics(labels, ensemble_predictions(logits[:, epoch], center_logits).tolist())
                 for epoch in range(logits.shape[1])]
    accuracies = [metrics["accuracy"] for metrics in per_epoch]
    best = accuracies.index(max(accuracies))
    return dict(final_epoch=per_epoch[-1], highest_epoch=dict(per_epoch[best], epoch=best + 1),
                test_accuracy_per_epoch=accuracies)
