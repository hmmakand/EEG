"""Adapt a trained model to one new subject using that subject's unlabeled trials.

Every function receives a loader over the held-out subject's trials and never
reads their labels. `adapt` changes the model in place and returns it, or
returns a wrapper for the prototype method.

    batchnorm   Recompute the BatchNorm statistics on the new subject
                (AdaBN, Li et al., 2016).
    infomax     After `batchnorm`, train the BatchNorm scale and shift so that
                predictions are confident per trial and balanced over the
                subject (information maximisation, Liang et al., 2020).
    selftrain   After `batchnorm`, fine-tune all weights on the model's own
                predictions for the subject, with logits centred so that both
                classes are used.
    prototype   Keep the model; classify each trial by cosine similarity of
                its features to two class prototypes built from the subject's
                most confident trials (T3A, Iwasawa & Matsuo, 2021).
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F

from .engine import predict

METHODS = ("none", "batchnorm", "infomax", "selftrain", "prototype")


def _batchnorm_layers(model):
    return [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def adapt_batchnorm(model, loader, device):
    model.eval()
    layers = _batchnorm_layers(model)
    momenta = [layer.momentum for layer in layers]
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None  # cumulative average over the subject's batches
        layer.train()
    for data in loader:
        predict(model, data.to(device))
    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum
    model.eval()
    return model


def _centred(logits):
    return logits - logits.mean(dim=0, keepdim=True)


def adapt_infomax(model, loader, device, epochs=10, learning_rate=1e-3):
    adapt_batchnorm(model, loader, device)
    parameters = [parameter for layer in _batchnorm_layers(model) for parameter in layer.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    model.eval()
    for _ in range(epochs):
        for data in loader:
            probability = predict(model, data.to(device)).softmax(dim=-1)
            entropy = -(probability * torch.log(probability + 1e-8)).sum(dim=-1).mean()
            mean = probability.mean(dim=0)
            diversity = -(mean * torch.log(mean + 1e-8)).sum()
            optimizer.zero_grad()
            (entropy - diversity).backward()
            optimizer.step()
    return model


def adapt_selftrain(model, loader, device, epochs=5, learning_rate=1e-4):
    adapt_batchnorm(model, loader, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.eval()  # dropout off; statistics stay those of the subject
    for _ in range(epochs):
        for data in loader:
            logits = predict(model, data.to(device))
            target = _centred(logits.detach()).argmax(dim=-1)
            optimizer.zero_grad()
            F.cross_entropy(logits, target).backward()
            optimizer.step()
    return model


class PrototypeClassifier(nn.Module):
    """Wraps a trained model; logits are similarities to subject-specific prototypes."""

    def __init__(self, model, prototypes, scale=10.0):
        super().__init__()
        self.model, self.scale = model, scale
        self.register_buffer("prototypes", prototypes)

    @property
    def uses_edge_attr(self):
        return getattr(self.model, "uses_edge_attr", False)

    @property
    def uses_signal(self):
        return getattr(self.model, "uses_signal", False)

    def forward(self, *inputs, **keywords):
        _, features = self.model(*inputs, return_features=True, **keywords)
        return self.scale * F.normalize(features, dim=-1) @ self.prototypes.T


@torch.no_grad()
def adapt_prototype(model, loader, device, keep=0.5):
    model.eval()
    logits, features = [], []
    for data in loader:
        output, feature = predict(model, data.to(device), return_features=True)
        logits.append(output)
        features.append(feature)
    logits, features = _centred(torch.cat(logits)), F.normalize(torch.cat(features), dim=-1)
    label, margin = logits.argmax(dim=-1), (logits[:, 1] - logits[:, 0]).abs()
    prototypes = []
    for target in range(logits.size(1)):
        members = (label == target).nonzero().view(-1)
        if len(members) == 0:
            raise ValueError("A class received no pseudo-labels; cannot build prototypes")
        chosen = members[margin[members].argsort(descending=True)[:max(1, int(len(members) * keep))]]
        prototypes.append(F.normalize(features[chosen].mean(dim=0), dim=-1))
    return PrototypeClassifier(model, torch.stack(prototypes)).to(device).eval()


def adapt(method, model, loader, device):
    """Return the model to evaluate with for `method`; labels in the loader are not used."""
    if method == "none":
        return model
    if method == "batchnorm":
        return adapt_batchnorm(model, loader, device)
    if method == "infomax":
        return adapt_infomax(model, loader, device)
    if method == "selftrain":
        return adapt_selftrain(model, loader, device)
    if method == "prototype":
        return adapt_prototype(model, loader, device)
    raise ValueError(f"Unknown adaptation method: {method}")


def adapted_copy(method, model, loader, device):
    """Like `adapt`, leaving `model` untouched."""
    return adapt(method, copy.deepcopy(model), loader, device)
