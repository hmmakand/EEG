"""Two-stage SPPM training loop (CFSPMNet paper, section 2.4).

Stage I: source-supervised pretraining + shared-prototype construction.
Stage II: joint source supervision and SPPM-calibrated target pseudo-supervision.

Ported from ``temp/sppm_strategy.py``. Two fixes versus the reference file:
the undefined external ``compute_metrics`` call is replaced with
``eeg_bci.tracking.metrics.compute_metrics`` (this project's multi-class-safe,
``labels=``-stable implementation), and ``logger.print(...)`` calls are
dropped -- notebooks handle printing/plotting directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_bci.cfspmnet.datasets import ArrayClassificationDataset, TargetPseudoDataset
from eeg_bci.cfspmnet.sppm import (
    accumulate_matching_stats,
    apply_confidence_pseudo_label_matching,
    apply_sppm_signature_prototype_matching,
    finalize_matching_stats,
    init_matching_aggregator,
)
from eeg_bci.tracking.metrics import compute_metrics


@dataclass(frozen=True)
class SPPMTrainingConfig:
    """SPPM hyperparameters. Defaults are the paper's XW-Stroke column (Table 2),
    since Liu2024 *is* XW-Stroke: alpha=0.98, pseudo_threshold=0.60."""

    device: torch.device
    alpha: float = 0.98
    pseudo_threshold: float = 0.60
    use_private_signature_matching: bool = True
    use_entropy_weight: bool = False
    static_pseudo: bool = False


def build_sppm_dataloaders(
    source_dataset: ArrayClassificationDataset,
    target_adapt_dataset: TargetPseudoDataset,
    target_test_dataset: ArrayClassificationDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Source, target-adaptation (shuffled), target-test, and target-init (unshuffled) loaders."""

    source_batch = min(batch_size, len(source_dataset))
    target_batch = min(batch_size, len(target_adapt_dataset))
    pin_memory = torch.cuda.is_available()

    source_loader = DataLoader(
        source_dataset, batch_size=source_batch, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    target_adapt_loader = DataLoader(
        target_adapt_dataset, batch_size=target_batch, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    target_test_loader = DataLoader(
        target_test_dataset, batch_size=target_batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    target_init_loader = DataLoader(
        target_adapt_dataset, batch_size=target_batch, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    return source_loader, target_adapt_loader, target_test_loader, target_init_loader


def sppm_matching_function(
    cfg: SPPMTrainingConfig,
    probabilities,
    signature_vectors,
    shared_prototypes,
    class_wise_matching_tolerance,
    existing_pseudo,
):
    if cfg.use_private_signature_matching:
        return apply_sppm_signature_prototype_matching(
            probabilities=probabilities,
            signature_vectors=signature_vectors,
            shared_prototypes=shared_prototypes,
            class_wise_matching_tolerance=class_wise_matching_tolerance,
            prob_threshold=cfg.pseudo_threshold,
            existing_pseudo=existing_pseudo,
        )
    return apply_confidence_pseudo_label_matching(
        probabilities=probabilities, prob_threshold=cfg.pseudo_threshold, existing_pseudo=existing_pseudo
    )


def train_source_only_epoch(source_loader: DataLoader, model, optimizer, device: torch.device) -> float:
    """Stage I: plain source-supervised training for one epoch."""

    model.train()
    total_loss = 0.0
    total_batches = len(source_loader)
    if total_batches == 0:
        return 0.0

    for x_s, y_s in source_loader:
        x_s = x_s.to(device, non_blocking=True)
        y_s = y_s.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_s), y_s)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / total_batches


def initialize_target_pseudo_labels(
    cfg: SPPMTrainingConfig, model, target_init_loader: DataLoader, shared_prototypes, class_wise_matching_tolerance
) -> tuple[float, dict]:
    """Populate initial pseudo-labels on the target-adaptation set before Stage II."""

    model.eval()
    total_accepted = 0
    total_samples = 0
    matching_agg = init_matching_aggregator()

    with torch.no_grad():
        for x_t, pseudo_y, signature_t, idx_t in target_init_loader:
            x_t = x_t.to(cfg.device, non_blocking=True)
            pseudo_y = pseudo_y.to(cfg.device, non_blocking=True)
            signature_t = signature_t.to(cfg.device, non_blocking=True)

            probs = torch.softmax(model(x_t), dim=1)
            matching_result = sppm_matching_function(
                cfg, probs, signature_t, shared_prototypes, class_wise_matching_tolerance, pseudo_y
            )
            target_dataset = cast(TargetPseudoDataset, target_init_loader.dataset)
            target_dataset.update_pseudo_labels(idx_t, matching_result["updated_pseudo"].cpu())
            total_accepted += int(matching_result["accepted_count"])
            total_samples += len(idx_t)
            accumulate_matching_stats(matching_agg, matching_result)

    return total_accepted / max(total_samples, 1), finalize_matching_stats(matching_agg)


def train_sppm_epoch(
    cfg: SPPMTrainingConfig,
    source_loader: DataLoader,
    target_loader: DataLoader,
    model,
    optimizer,
    shared_prototypes,
    class_wise_matching_tolerance,
) -> dict:
    """Stage II: one epoch of joint source supervision + SPPM-calibrated target pseudo-supervision."""

    model.train()
    total_loss_epoch = 0.0
    total_source_loss = 0.0
    total_target_loss = 0.0
    total_batches = len(source_loader)
    matching_agg = init_matching_aggregator()

    if total_batches == 0:
        return {
            "train_loss": 0.0,
            "source_loss": 0.0,
            "target_loss": 0.0,
            "accepted_ratio": 0.0,
            "active_ratio": 0.0,
            "matching_stats": finalize_matching_stats(matching_agg),
        }

    target_dataset = cast(TargetPseudoDataset, target_loader.dataset)
    target_iter = iter(target_loader)
    total_target_samples = 0

    for x_s, y_s in source_loader:
        try:
            x_t, tgt_pseudo_y, signature_t, tgt_idx = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            x_t, tgt_pseudo_y, signature_t, tgt_idx = next(target_iter)

        x_s = x_s.to(cfg.device, non_blocking=True)
        y_s = y_s.to(cfg.device, non_blocking=True)
        x_t = x_t.to(cfg.device, non_blocking=True)
        tgt_pseudo_y = tgt_pseudo_y.to(cfg.device, non_blocking=True)
        signature_t = signature_t.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss_source = F.cross_entropy(model(x_s), y_s)

        tgt_logits = model(x_t)
        log_probs = F.log_softmax(tgt_logits, dim=1)
        target_weights = 1.0
        if cfg.use_entropy_weight:
            tgt_probs = torch.softmax(tgt_logits, dim=1)
            entropy = -torch.sum(tgt_probs * torch.log(tgt_probs + 1e-10), dim=1)
            target_weights = torch.exp(-entropy).detach()
        loss_target_sample = -torch.sum(tgt_pseudo_y * log_probs, dim=1)
        loss_target = torch.mean(
            target_weights * loss_target_sample if torch.is_tensor(target_weights) else loss_target_sample
        )

        total_loss = cfg.alpha * loss_source + (1.0 - cfg.alpha) * loss_target
        total_loss.backward()
        optimizer.step()

        if not cfg.static_pseudo:
            # Refresh with inference semantics: otherwise dropout stays active and
            # BatchNorm statistics are updated a second time on every target batch.
            model.eval()
            with torch.no_grad():
                updated_probs = torch.softmax(model(x_t), dim=1)
                matching_result = sppm_matching_function(
                    cfg, updated_probs, signature_t, shared_prototypes, class_wise_matching_tolerance, tgt_pseudo_y
                )
                target_dataset.update_pseudo_labels(tgt_idx, matching_result["updated_pseudo"].cpu())
                accumulate_matching_stats(matching_agg, matching_result)
            model.train()

        total_target_samples += len(tgt_idx)
        total_loss_epoch += total_loss.item()
        total_source_loss += loss_source.item()
        total_target_loss += loss_target.item()

    matching_stats = finalize_matching_stats(matching_agg)
    return {
        "train_loss": total_loss_epoch / total_batches,
        "source_loss": total_source_loss / total_batches,
        "target_loss": total_target_loss / total_batches,
        "accepted_ratio": matching_stats["accepted_count"] / max(total_target_samples, 1),
        "active_ratio": target_dataset.active_ratio(),
        "matching_stats": matching_stats,
    }


def evaluate_target(model, target_loader: DataLoader, device: torch.device, n_outputs: int) -> dict:
    """Evaluate on labeled target data after the selected protocol finishes."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true_list, y_pred_list, y_prob_list = [], [], []

    with torch.no_grad():
        for x_t, y_t in target_loader:
            x_t = x_t.to(device, non_blocking=True)
            y_t = y_t.to(device, non_blocking=True)
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            total_loss += F.cross_entropy(logits, y_t, reduction="sum").item()
            total_samples += y_t.size(0)
            y_true_list.append(y_t.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())
            y_prob_list.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob, labels=list(range(n_outputs)))
    return {
        "test_loss": total_loss / max(total_samples, 1),
        "n_test_windows": total_samples,
        "metrics": metrics,
    }
