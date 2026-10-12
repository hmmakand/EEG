"""Calibration on the unlabelled test subject (PGTC of PA-TCNet, Wang et al. 2026).

After a warm-up on the training subjects, the model's own predictions for the
test subject's trials are used as extra training targets. A prediction
becomes a target only if

  - the model is confident: its probability is at least `calibration_confidence`, and
  - the motor areas agree: the trial's motor-area features (data/roi.py) are
    similar enough to the template of the predicted class. A template is the
    mean feature vector of that class over the training subjects; the
    threshold of a class is the mean minus one standard deviation of the
    similarities of its own training trials, and at least
    `calibration_similarity_floor`.

The loss becomes  w * CE(training batch) + (1 - w) * CE(test batch, targets),
with w = `calibration_source_weight`; test trials without a target contribute
zero. The targets are refreshed after every epoch. A trial keeps its target
until a later prediction that passes both checks replaces it.

No label of the test subject is used.

Differences from the published code, chosen to fit this pipeline:
  - The motor-area features are band powers standardised per subject, not raw
    signals averaged over electrodes.
  - Targets are refreshed once per epoch from all test trials, with the
    logits centred as in the final decision, not after every batch.
"""
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from .engine import predict


def class_templates(features, labels, floor: float):
    """(templates [classes, features], thresholds [classes]) from the training subjects' trials."""
    features = F.normalize(features, dim=1)
    templates, thresholds = [], []
    for label in (0, 1):
        own = features[labels == label]
        if len(own) == 0:
            raise ValueError(f"No training trial of class {label}")
        template = F.normalize(own.mean(dim=0), dim=0)
        similarity = own @ template
        templates.append(template)
        thresholds.append(max(float(similarity.mean() - similarity.std()), floor))
    return torch.stack(templates), torch.tensor(thresholds)


def calibrated_targets(logits, features, templates, thresholds, confidence: float, center_logits: bool, existing):
    """New targets [trials, classes]; rows that pass neither check keep their `existing` value."""
    if center_logits:
        logits = logits - logits.mean(dim=0, keepdim=True)
    probability, prediction = logits.softmax(dim=1).max(dim=1)
    similarity = (F.normalize(features, dim=1) * templates[prediction]).sum(dim=1)
    accepted = (probability >= confidence) & (similarity >= thresholds[prediction])
    targets = existing.clone()
    targets[accepted] = F.one_hot(prediction[accepted], num_classes=logits.shape[1]).float()
    return targets


class TargetCalibration:
    """The test subject's unlabelled trials, the templates, and the current targets of one model."""

    def __init__(self, training, testing, config):
        for graph in (training[0], testing[0]):
            if getattr(graph, "roi", None) is None:
                raise ValueError("Target calibration needs motor-area features; the dataset has no channel names")
        labels = torch.stack([graph.y for graph in training])
        self.templates, self.thresholds = class_templates(torch.cat([g.roi for g in training]), labels,
                                                          config.calibration_similarity_floor)
        self.features = torch.cat([graph.roi for graph in testing])
        self.config = config
        self.graphs = []
        for index, graph in enumerate(testing):
            copy = graph.clone()
            copy.trial_number = torch.tensor(index)
            del copy.y                                  # the labels of the test subject are not available here
            self.graphs.append(copy)
        self.targets = torch.zeros(len(testing), 2)

    def loader(self):
        batch = self.config.batch_size
        return DataLoader(self.graphs, batch_size=batch, shuffle=True, drop_last=len(self.graphs) % batch == 1)

    def refresh(self, logits) -> None:
        """Update the targets from the model's current logits of all test trials (in their original order)."""
        self.targets = calibrated_targets(logits, self.features, self.templates, self.thresholds,
                                          self.config.calibration_confidence, self.config.center_logits, self.targets)

    def summary(self, labels) -> dict:
        """Share of test trials with a target, and how many targets are right (a diagnostic; uses the labels)."""
        labelled = self.targets.sum(dim=1) > 0
        correct = self.targets.argmax(dim=1)[labelled] == torch.tensor(labels)[labelled]
        return dict(labelled_fraction=float(labelled.float().mean()),
                    target_accuracy=float(correct.float().mean()) if labelled.any() else None)


def train_epoch_calibrated(model, loader, calibration: TargetCalibration, optimizer, criterion, device) -> None:
    """One epoch on the training subjects, with a batch of the test subject's trials at every step."""
    weight = calibration.config.calibration_source_weight
    target_loader = calibration.loader()
    target_batches = iter(target_loader)
    model.train()
    for data in loader:
        try:
            target = next(target_batches)
        except StopIteration:
            target_batches = iter(target_loader)
            target = next(target_batches)
        data, target = data.to(device), target.to(device)
        source_loss = criterion(predict(model, data), data.y)
        wanted = calibration.targets[target.trial_number.cpu()].to(device)
        target_loss = -(wanted * F.log_softmax(predict(model, target), dim=1)).sum(dim=1).mean()
        loss = weight * source_loss + (1.0 - weight) * target_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
