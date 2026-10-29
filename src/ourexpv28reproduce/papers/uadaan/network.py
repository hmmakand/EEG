"""UA-DAAN, ported from the reference implementation (original/bayesian.py, extractor.py, classifier.py,
discriminator.py, model.py); only the input size is set per dataset (the reference has 640 = 128 x 5).

    feature extractor G_f   BatchNorm(input) -> Linear 256 -> Dropout 0.2 -> ReLU -> Linear 64 -> Dropout -> ReLU
                            -> Linear 16 -> Dropout                       (MC Dropout: also at test time)
    label classifier G_y    ReLU -> BayesLinear 4 -> Tanh -> BayesLinear 2 -> BatchNorm(2) -> Softmax
                            (weights drawn from N(mu, exp(log_sigma)^2) at every forward pass; prior N(0, 0.1^2))
    discriminators          ReLU -> Linear 4 -> ReLU -> Linear 2 -> Softmax: one global (on f), one local per class
                            (on p_c * f), behind a gradient reversal layer with the DANN schedule
    domain loss             entropy of the discriminator's output (Eq. 10), weighted by 1 + exp(-H(p)) (Eq. 9),
                            where H(p) is the entropy of the class prediction
    dynamic factor          from the global and local A-distances of the previous epoch (Eq. 13-15)

Every prediction averages 10 Monte Carlo forward passes. Kept as the reference does it, although unusual:
the cross-entropy is applied to the averaged probabilities (nn.CrossEntropyLoss on softmax outputs); the
domain labels are not used by the domain loss (it is the entropy of Eq. 10).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from .settings import ModelSettings, TrainingSettings

CLASSES = 2


class BayesLinear(nn.Module):
    def __init__(self, prior_mu: float, prior_sigma: float, in_features: int, out_features: int):
        super().__init__()
        self.prior_mu, self.prior_sigma, self.prior_log_sigma = prior_mu, prior_sigma, math.log(prior_sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_log_sigma = nn.Parameter(torch.empty(out_features))
        bound = 1 / math.sqrt(in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def forward(self, x):
        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
        return F.linear(x, weight, bias)


def _kl(mu_0, log_sigma_0, mu_1, log_sigma_1):
    return (log_sigma_1 - log_sigma_0 + (torch.exp(log_sigma_0) ** 2 + (mu_0 - mu_1) ** 2)
            / (2 * math.exp(log_sigma_1) ** 2) - 0.5).sum()


def bayesian_kl(model):
    """bayesian.BKLLoss(reduction='mean'): the KL to the prior, summed over the Bayesian layers' weights and
    biases, divided by their number."""
    total, count = 0.0, 0
    for m in model.modules():
        if isinstance(m, BayesLinear):
            total = total + _kl(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
            total = total + _kl(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
            count += m.weight_mu.numel() + m.bias_mu.numel()
    return total / count


def extractor(inputs: int, s: ModelSettings):
    h1, h2, h3 = s.hidden
    return nn.Sequential(nn.BatchNorm1d(inputs), nn.Linear(inputs, h1), nn.Dropout(s.dropout), nn.ReLU(),
                         nn.Linear(h1, h2), nn.Dropout(s.dropout), nn.ReLU(), nn.Linear(h2, h3), nn.Dropout(s.dropout))


def classifier(s: ModelSettings):
    return nn.Sequential(nn.ReLU(), BayesLinear(s.prior_mu, s.prior_sigma, s.hidden[-1], s.classifier_hidden),
                         nn.Tanh(), BayesLinear(s.prior_mu, s.prior_sigma, s.classifier_hidden, CLASSES),
                         nn.BatchNorm1d(CLASSES), nn.Softmax(1))


def discriminator(s: ModelSettings):
    return nn.Sequential(nn.ReLU(), nn.Linear(s.hidden[-1], s.discriminator_hidden), nn.ReLU(),
                         nn.Linear(s.discriminator_hidden, 2), nn.Softmax(dim=1))


class ReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def entropy(probabilities):
    """-sum p log p per row, NaN set to 0 (the reference's get_uncertainty)."""
    return torch.nan_to_num((probabilities * torch.log(probabilities)).sum(dim=1).neg(), 0.0)


def uncertain_loss(domain_probabilities, uncertainty):
    """discriminator.UncertainLoss: the discriminator's entropy, plus the same weighted by exp(-uncertainty)."""
    loss = entropy(domain_probabilities + 1e-10)
    weighted = torch.nan_to_num(loss * torch.exp(uncertainty.neg()), 0.0)
    return (loss + weighted).mean()


class DynamicAdversarialLoss(nn.Module):
    """discriminator.DAANloss: global and local uncertainty-aware adversarial losses and the dynamic factor."""

    def __init__(self, s: ModelSettings, t: TrainingSettings):
        super().__init__()
        self.global_discriminator = discriminator(s)
        self.local_discriminators = nn.ModuleList([discriminator(s) for _ in range(CLASSES)])
        self.t = t
        self.iteration = 0
        self.global_distance, self.local_distance, self.dynamic_factor = 0.0, 0.0, 0.5

    def reversal_weight(self):
        p = self.iteration / self.t.reversal_iterations
        return 2.0 / (1.0 + np.exp(-self.t.reversal_gamma * p)) - 1

    def forward(self, source, target, source_probabilities, target_probabilities):
        lamb = self.reversal_weight()
        self.iteration = min(self.iteration + 1, self.t.reversal_iterations)
        source_u, target_u = entropy(source_probabilities), entropy(target_probabilities)
        global_loss = 0.5 * (self.global_term(source, source_u, lamb) + self.global_term(target, target_u, lamb)) \
            * self.t.global_scale
        local_loss = 0.5 * (self.local_term(source, source_probabilities, source_u, lamb)
                            + self.local_term(target, target_probabilities, target_u, lamb)) * self.t.local_scale
        self.global_distance += 2 * (1 - 2 * global_loss.item())
        self.local_distance += 2 * (1 - 2 * (local_loss / 2).item())
        return (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * local_loss

    def global_term(self, x, uncertainty, lamb):
        return uncertain_loss(self.global_discriminator(ReverseLayer.apply(x, lamb)), uncertainty)

    def local_term(self, x, probabilities, uncertainty, lamb):
        x = ReverseLayer.apply(x, lamb)
        return sum(uncertain_loss(d(probabilities[:, c:c + 1] * x), uncertainty)
                   for c, d in enumerate(self.local_discriminators))

    def update_dynamic(self, batches: int) -> None:
        """At the start of every epoch, from the A-distances accumulated in the previous one (update_dynamic)."""
        if self.global_distance == 0 and self.local_distance == 0:
            self.dynamic_factor = 0.5
        else:
            g, l = self.global_distance / batches, self.local_distance / batches
            self.dynamic_factor = 1 - g / (g + l)
        self.global_distance, self.local_distance = 0.0, 0.0


class UADAAN(nn.Module):
    def __init__(self, inputs: int, model_settings: ModelSettings = ModelSettings(),
                 training: TrainingSettings = TrainingSettings()):
        super().__init__()
        self.feature_extractor = extractor(inputs, model_settings)
        self.label_clf = classifier(model_settings)
        self.adapt_loss = DynamicAdversarialLoss(model_settings, training)
        self.samples = model_settings.monte_carlo
        self.t = training

    def sampled(self, x):
        """Mean features and mean class probabilities over the Monte Carlo passes."""
        features, probabilities = [], []
        for _ in range(self.samples):
            f = self.feature_extractor(x)
            features.append(f)
            probabilities.append(self.label_clf(f))
        return sum(features) / len(features), sum(probabilities) / len(probabilities)

    def forward(self, source, target, source_label):
        """(classification loss, transfer loss) as model.Model.forward; the training loss is their difference."""
        source_f, source_p = self.sampled(source)
        source_p = source_p + 1e-10
        loss = (self.t.classification_weight * F.cross_entropy(source_p, source_label)
                + self.t.kl_weight * bayesian_kl(self.label_clf))
        target_f, target_p = self.sampled(target)
        return loss, self.adapt_loss(source_f, target_f, source_p, target_p + 1e-10)

    def predict(self, x):
        return self.sampled(x)[1]


def build_model(inputs: int, model_settings: ModelSettings = ModelSettings(),
                training: TrainingSettings = TrainingSettings()) -> UADAAN:
    return UADAAN(inputs, model_settings, training)
