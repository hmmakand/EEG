"""One training epoch, the logits of the test samples, and the decision of the ensemble."""
import torch


def predict(model, data):
    return model(data.x, data.edge_index, data.batch, data.edge_attr, data.signal)


def train_epoch(model, loader, optimizer, criterion, device):
    """One epoch; returns (mean training loss, training accuracy) over the epoch's batches."""
    model.train()
    total, correct, count = 0.0, 0, 0
    for data in loader:
        data = data.to(device)
        logits = predict(model, data)
        loss = criterion(logits, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(data.y)
        correct += int((logits.argmax(dim=1) == data.y).sum())
        count += len(data.y)
    return total / count, correct / count


@torch.no_grad()
def test_logits(model, batches, device) -> torch.Tensor:
    """Logits [samples, classes] of the test batches. Draws no random numbers, so
    calling it between epochs does not alter training."""
    model.eval()
    return torch.cat([predict(model, data.to(device)) for data in batches]).cpu()


def ensemble_scores(member_logits: torch.Tensor, center_logits: bool):
    """(predicted classes [samples], probability of class 1 [samples]) from the logits of the models
    [models, samples, classes]; the decision of ensemble_predictions."""
    scores = member_logits[0] if len(member_logits) == 1 else member_logits.softmax(dim=-1).mean(dim=0).log()
    if center_logits:
        scores = scores - scores.mean(dim=0, keepdim=True)
    return scores.argmax(dim=1), scores.softmax(dim=-1)[:, 1]


def ensemble_predictions(member_logits: torch.Tensor, center_logits: bool) -> torch.Tensor:
    """Predicted classes [samples] from the logits of the models [models, samples, classes].

    The class probabilities of the models are averaged. `center_logits`
    subtracts the mean over the evaluated samples before the decision. It
    removes a subject-wide bias toward one class without using labels, and
    assumes the samples belong to one subject.
    """
    scores = member_logits[0] if len(member_logits) == 1 else member_logits.softmax(dim=-1).mean(dim=0).log()
    if center_logits:
        scores = scores - scores.mean(dim=0, keepdim=True)
    return scores.argmax(dim=1)
