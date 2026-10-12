"""Training losses that discourage subject-specific features.

Both act on the model's learned per-graph features (`return_features=True`)
and are added to the classification loss during training only.

    adversarial   A subject classifier is trained on the features through a
                  gradient-reversal layer (Ganin & Lempitsky, 2015): it learns
                  to tell the training subjects apart while the model learns
                  features from which it cannot.
    contrastive   Supervised contrastive loss (Khosla et al., 2020) whose
                  positives are trials of the same class from OTHER subjects,
                  so a class is pulled together across subjects.
    proxy         Proxy-Anchor metric learning (Kim et al., 2020): one
                  learned proxy per class; trials are pulled to their class
                  proxy and pushed from the other, whatever their subject.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(context, values, scale):
        context.scale = scale
        return values.view_as(values)

    @staticmethod
    def backward(context, gradient):
        return -context.scale * gradient, None


class SubjectInvariance(nn.Module):
    """`forward(features, labels, subjects, progress)` returns the extra loss.

    `subjects` holds the original subject numbers; `progress` in [0, 1] is the
    fraction of training completed and ramps the adversarial strength up.
    """

    def __init__(self, mode, feature_dim, subject_ids, weight, hidden=64, projection=32, temperature=0.1,
                 num_classes=2):
        super().__init__()
        if mode not in ("adversarial", "contrastive", "proxy"):
            raise ValueError(f"Unknown invariance mode: {mode}")
        if weight < 0 or not math.isfinite(weight):
            raise ValueError("invariance weight must be non-negative and finite")
        subject_ids = sorted(set(int(subject) for subject in subject_ids))
        if len(subject_ids) < 2:
            raise ValueError("Subject invariance needs at least two training subjects")
        self.mode, self.weight, self.temperature = mode, weight, temperature
        lookup = torch.full((max(subject_ids) + 1,), -1, dtype=torch.long)
        lookup[subject_ids] = torch.arange(len(subject_ids))
        self.register_buffer("lookup", lookup)
        self.norm = nn.BatchNorm1d(feature_dim)
        output = len(subject_ids) if mode == "adversarial" else projection
        self.head = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, output))
        if mode == "proxy":
            self.proxies = nn.Parameter(torch.randn(num_classes, projection))
            self.margin, self.alpha = 0.1, 32.0

    def forward(self, features, labels, subjects, progress):
        subjects = self.lookup[subjects.to(self.lookup.device).long().view(-1)]
        if (subjects < 0).any():
            raise ValueError("A batch contains a subject that is not a training subject")
        features = self.norm(features)
        if self.mode == "adversarial":
            # Standard ramp: 0 at the start of training, close to 1 at the end.
            strength = self.weight * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            return F.cross_entropy(self.head(_ReverseGradient.apply(features, strength)), subjects)
        embedding = F.normalize(self.head(features), dim=-1)
        if self.mode == "proxy":
            cosine = embedding @ F.normalize(self.proxies, dim=-1).T  # [trials, classes]
            own = F.one_hot(labels.view(-1), cosine.size(1)).bool()
            pull = torch.where(own, torch.exp(-self.alpha * (cosine - self.margin)), torch.zeros_like(cosine)).sum(dim=0)
            push = torch.where(~own, torch.exp(self.alpha * (cosine + self.margin)), torch.zeros_like(cosine)).sum(dim=0)
            present = own.any(dim=0)
            return self.weight * (torch.log1p(pull)[present].mean() + torch.log1p(push).mean())
        similarity = embedding @ embedding.T / self.temperature
        own = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
        log_probability = similarity.masked_fill(own, float("-inf")).log_softmax(dim=-1)
        positive = (labels.view(-1, 1) == labels.view(1, -1)) & (subjects.view(-1, 1) != subjects.view(1, -1))
        anchors = positive.any(dim=-1)
        if not anchors.any():
            return features.sum() * 0.0
        per_anchor = (log_probability.masked_fill(~positive, 0.0).sum(dim=-1)[anchors]
                      / positive.sum(dim=-1)[anchors])
        return -self.weight * per_anchor.mean()
