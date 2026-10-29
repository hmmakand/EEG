"""PA-TCNet's leave-one-subject-out training, ported from the published train.py.

One fold: the held-out subject is the target domain. The model trains on the
other subjects alone for the warm-up epochs; afterwards every step also trains
on a batch of the target subject's trials, with targets set by PGTC
(pgtc.py): a prediction becomes a target if it is confident and, where the
channel names are known, if its ROI vector matches the template of the
predicted class. The loss is alpha * source + (1 - alpha) * target.

Kept as in the published code:
  - Adam, learning rate 0.001, weight decay 0.001, batch 64, shuffled loaders;
  - the targets are initialised after the warm-up (evaluation mode, all target
    trials) and refreshed after every step on that step's batch, by a forward
    pass in training mode, as the code does;
  - after every epoch the held-out subject is evaluated.

The published code then stops training when the loss of the held-out subject
has not improved for 30 epochs, and reports the epoch with the highest
accuracy on the held-out subject. That choice is made with the test labels.
Here every fold runs all 200 epochs and keeps the evaluation of every epoch;
from these, both scores are reconstructed (see fold_scores()):

  as_published  the highest accuracy up to the epoch at which the published
                code would have stopped (the code's protocol)
  final_epoch   the score after the last epoch, chosen without test labels
                (fixed number of epochs; our protocol)

Training before the stopping epoch is shared; the missing utility implementations prevent an exact checkpoint-selection guarantee.

Recorded for reporting, without changing the training: the loss and accuracy
on the source batches per epoch (training mode), the test loss and accuracy
per epoch, the share of target trials with a calibrated target and how many
of those targets are right (a diagnostic with the test labels), the
predictions and probabilities of every test trial for both scores (ROC
curves), the training time and the peak GPU memory of the fold.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ...evaluation.complexity import synchronize
from ...evaluation.metrics import classification_metrics
from .network import build_model
from .original.pgtc import apply_confidence_calibration, apply_pgtc_calibration
from .original.physiology import build_class_templates, compute_roi_features
from .settings import DatasetSettings, ModelSettings, TrainingSettings


def loader(tensors, batch_size, shuffle):
    """Drop singleton batches only in training; evaluation and calibration visit every trial."""
    size = min(batch_size, len(tensors[0]))
    return DataLoader(TensorDataset(*tensors), batch_size=size, shuffle=shuffle,
                      drop_last=shuffle and len(tensors[0]) > 1 and len(tensors[0]) % size == 1)


class Calibration:
    """The target subject's trials, its ROI vectors and the current calibrated targets (PGTC)."""

    def __init__(self, x_source, y_source, x_target, channel_info, settings: DatasetSettings, device):
        self.settings, self.device = settings, device
        self.targets = torch.zeros(len(x_target), 2)
        if channel_info is None:
            # No channel names (jia2019): the confidence check of pgtc.py alone.
            self.templates = self.thresholds = self.roi = None
        else:
            source_roi = compute_roi_features(x_source, channel_info)["roi_vector"]
            templates, thresholds = build_class_templates(source_roi, y_source, num_classes=2,
                                                          floor=settings.roi_floor)
            self.templates = torch.tensor(templates, device=device)
            self.thresholds = torch.tensor(thresholds, device=device)
            self.roi = torch.tensor(compute_roi_features(x_target, channel_info)["roi_vector"])

    def calibrate(self, probabilities, indices):
        """Update the targets of the trials `indices` from the model's probabilities."""
        existing = self.targets[indices].to(self.device)
        if self.templates is None:
            result = apply_confidence_calibration(probabilities, self.settings.confidence, existing)
        else:
            result = apply_pgtc_calibration(probabilities, self.roi[indices].to(self.device), self.templates,
                                            self.thresholds, self.settings.confidence, existing)
        self.targets[indices] = result["updated_targets"].cpu()
        return int(result["accepted_mask"].sum())

    def active_fraction(self) -> float:
        return float((self.targets.sum(dim=1) > 0).float().mean())

    def target_accuracy(self, labels) -> float | None:
        """Share of the calibrated targets that equal the true label (a diagnostic; uses the test labels)."""
        active = self.targets.sum(dim=1) > 0
        if not active.any():
            return None
        return float((self.targets.argmax(dim=1)[active] == torch.as_tensor(labels)[active]).float().mean())


def evaluate(model, test_loader, device):
    """(loss, labels, predictions, probability of class 1) on the held-out subject."""
    model.eval()
    losses, labels, predictions, scores = 0.0, [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses += float(F.cross_entropy(logits, y, reduction="sum"))
            labels += y.tolist()
            predictions += logits.argmax(dim=1).tolist()
            scores += logits.softmax(dim=1)[:, 1].tolist()
    return losses / max(len(labels), 1), labels, predictions, scores


def run_fold(x_source, y_source, x_target, y_target, channel_info, settings: DatasetSettings, device,
             model_settings: ModelSettings = ModelSettings(), training: TrainingSettings = TrainingSettings(),
             description="", show_progress: bool = True) -> dict:
    """Train on the source subjects, adapt to the target subject, and evaluate it after every epoch."""
    model = build_model(settings.dataset, model_settings, settings=settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    source_loader = loader((torch.tensor(x_source), torch.tensor(y_source)), training.batch_size, shuffle=True)
    indices = torch.arange(len(x_target))
    target_loader = loader((torch.tensor(x_target), indices), training.batch_size, shuffle=True)
    target_init_loader = loader((torch.tensor(x_target), indices), training.batch_size, shuffle=False)
    test_loader = loader((torch.tensor(x_target), torch.tensor(y_target)), training.batch_size, shuffle=False)
    calibration = Calibration(x_source, y_source, x_target, channel_info, settings, device)

    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    for epoch in tqdm(range(1, training.epochs + 1), desc=description, leave=False, disable=not show_progress):
        model.train()
        train_loss, train_correct, train_count = 0.0, 0, 0
        if epoch <= settings.warmup_epochs:
            for x, y in source_loader:
                y = y.to(device)
                logits = model(x.to(device))
                loss = F.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
                train_correct += int((logits.argmax(dim=1) == y).sum())
                train_count += len(y)
        else:
            if epoch == settings.warmup_epochs + 1:                   # initialise the targets (evaluation mode)
                model.eval()
                with torch.no_grad():
                    for x, index in target_init_loader:
                        calibration.calibrate(torch.softmax(model(x.to(device)), dim=1), index)
                model.train()
            target_batches = iter(target_loader)
            for x_s, y_s in source_loader:
                try:
                    x_t, index = next(target_batches)
                except StopIteration:
                    target_batches = iter(target_loader)
                    x_t, index = next(target_batches)
                x_s, y_s, x_t = x_s.to(device), y_s.to(device), x_t.to(device)
                logits = model(x_s)
                loss_source = F.cross_entropy(logits, y_s)
                wanted = calibration.targets[index].to(device)
                loss_target = torch.mean(-torch.sum(wanted * F.log_softmax(model(x_t), dim=1), dim=1))
                loss = settings.source_weight * loss_source + (1.0 - settings.source_weight) * loss_target
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += loss_source.item() * len(y_s)
                train_correct += int((logits.argmax(dim=1) == y_s).sum())
                train_count += len(y_s)
                with torch.no_grad():                                  # as the code: training mode
                    calibration.calibrate(torch.softmax(model(x_t), dim=1), index)
        loss, labels, predictions, scores = evaluate(model, test_loader, device)
        history.append(dict(epoch=epoch, loss=loss, accuracy=float(np.mean(np.array(labels) == np.array(predictions))),
                            predictions=predictions, scores=scores, calibrated_fraction=calibration.active_fraction(),
                            target_accuracy=calibration.target_accuracy(y_target),
                            train_loss=train_loss / train_count, train_accuracy=train_correct / train_count))
    synchronize(device)
    seconds = time.perf_counter() - start
    per_epoch = lambda key: [h[key] for h in history]
    return dict(fold_scores(history, labels, training.patience), labels=labels,
                train_loss_per_epoch=per_epoch("train_loss"), train_accuracy_per_epoch=per_epoch("train_accuracy"),
                test_accuracy_per_epoch=per_epoch("accuracy"), test_loss_per_epoch=per_epoch("loss"),
                calibrated_fraction_per_epoch=per_epoch("calibrated_fraction"),
                target_accuracy_per_epoch=per_epoch("target_accuracy"),
                training_seconds=seconds,
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                training_trials=len(x_source), testing_trials=len(x_target))


def stopping_epoch(losses, patience: int) -> int:
    """The epoch at which the published code stops: the loss of the held-out subject has not improved
    for `patience` epochs (EarlyStopping is not published; this is its usual definition)."""
    best, waited = float("inf"), 0
    for epoch, loss in enumerate(losses, start=1):
        if loss < best:
            best, waited = loss, 0
        else:
            waited += 1
            if waited >= patience:
                return epoch
    return len(losses)


def fold_scores(history, labels, patience: int) -> dict:
    """The two scores of a fold, from the evaluations after every epoch, with the predictions and the
    probabilities of class 1 of every test trial behind each."""
    stop = stopping_epoch([h["loss"] for h in history], patience)
    # BestModelSaver(monitor="acc") keeps the first epoch with the highest accuracy, up to the stop.
    accuracies = [h["accuracy"] for h in history[:stop]]
    best = int(np.argmax(accuracies))
    final = history[-1]
    return dict(final_epoch=dict(classification_metrics(labels, final["predictions"], final["scores"]),
                                 epoch=len(history)),
                as_published=dict(classification_metrics(labels, history[best]["predictions"], history[best]["scores"]),
                                  epoch=best + 1, stopping_epoch=stop),
                final_epoch_predictions=final["predictions"], final_epoch_scores=final["scores"],
                as_published_predictions=history[best]["predictions"], as_published_scores=history[best]["scores"])
