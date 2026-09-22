"""FixMatch-style semi-supervised training on graph node features with fixed connectivity."""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@contextmanager
def _deterministic_view(model):
    # eval also disables functional dropout used by GAT.forward.
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def validate_fixmatch_settings(lambda_u, confidence_threshold, weak_noise_std,
                               strong_noise_std, strong_mask_prob):
    for name, value in (("lambda_u", lambda_u), ("weak_noise_std", weak_noise_std),
                       ("strong_noise_std", strong_noise_std)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(confidence_threshold) or not 0 < confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be finite and in (0, 1]")
    if not math.isfinite(strong_mask_prob) or not 0 <= strong_mask_prob < 1:
        raise ValueError("strong_mask_prob must be finite and in [0, 1)")


def _add_gaussian_noise(x, std):
    if std == 0:
        return x
    return x + std * torch.randn_like(x)


def _mask_features(x, prob):
    if prob == 0:
        return x
    keep = (torch.rand_like(x) >= prob).to(x.dtype)
    return x * keep


@dataclass(frozen=True)
class FixMatchMetrics:
    """Loss averages weighted by the number of labeled graphs per update."""
    supervised_loss: float
    unsupervised_loss: float
    total_loss: float
    pseudo_label_mask_rate: float


def train_epoch_fixmatch(model, loader, optimizer, device, *, unlabeled_loader=None,
                         lambda_u=1.0, confidence_threshold=0.95,
                         weak_noise_std=0.05, strong_noise_std=0.2,
                         strong_mask_prob=0.3) -> FixMatchMetrics:
    """Train for one labeled-loader epoch, optionally cycling unlabeled batches.

    Labeled graphs are weakly augmented (small Gaussian noise) before the
    supervised cross-entropy loss. Each unlabeled batch gets a pseudo-label from
    its own weakly augmented, dropout-disabled prediction; graphs whose top
    class probability is below confidence_threshold are excluded. A strongly
    augmented view (larger Gaussian noise plus random feature masking) of the
    surviving graphs is then trained toward that pseudo-label with
    cross-entropy, weighted by lambda_u. Labels on unlabeled batches are never
    read. Supply a reiterable, nonempty unlabeled loader containing only
    training data. Zero lambda_u or a missing unlabeled_loader skips all
    pseudo-labeling work.
    """
    validate_fixmatch_settings(lambda_u, confidence_threshold, weak_noise_std,
                               strong_noise_std, strong_mask_prob)
    model.train()
    unlabeled_iterator = iter(unlabeled_loader) if unlabeled_loader is not None and lambda_u > 0 else None
    totals = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(_add_gaussian_noise(data.x, weak_noise_std), data.edge_index, data.batch)
        supervised = F.cross_entropy(logits, data.y)
        unsupervised = supervised.new_zeros(())
        mask_rate = supervised.new_zeros(())
        size = int(logits.shape[0])
        if unlabeled_iterator is not None and unlabeled_loader is not None:
            extra = next(unlabeled_iterator, None)
            if extra is None:
                unlabeled_iterator = iter(unlabeled_loader)
                extra = next(unlabeled_iterator, None)
            if extra is None:
                raise ValueError("unlabeled_loader must be nonempty and reiterable")
            extra = extra.to(device)
            with _deterministic_view(model), torch.no_grad():
                weak_prob = torch.softmax(
                    model(_add_gaussian_noise(extra.x, weak_noise_std), extra.edge_index, extra.batch),
                    dim=1)
            confidence, pseudo_label = weak_prob.max(dim=1)
            mask = confidence >= confidence_threshold
            mask_rate = mask.float().mean()
            if mask.any():
                strong_x = _mask_features(_add_gaussian_noise(extra.x, strong_noise_std), strong_mask_prob)
                strong_logits = model(strong_x, extra.edge_index, extra.batch)
                unsupervised = F.cross_entropy(strong_logits[mask], pseudo_label[mask])
        loss = supervised + lambda_u * unsupervised
        loss.backward()
        optimizer.step()
        for index, value in enumerate((supervised, unsupervised, loss, mask_rate)):
            totals[index] += float(value.detach()) * size
        count += size
    if count == 0:
        raise ValueError("Cannot train on an empty labeled loader")
    return FixMatchMetrics(*(value / count for value in totals))
