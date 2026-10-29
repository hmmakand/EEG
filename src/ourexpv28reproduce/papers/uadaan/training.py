"""UA-DAAN training of one fold, as the reference's ua-daan12fold.py does it.

Source domain: the fold's labelled training samples (batches of 98, shuffled).
Target domain: the fold's test samples without labels (batches of 8, in
order, cycled). Every epoch first updates the dynamic adversarial factor
from the previous epoch, then takes one step per source batch with the next
target batch; the loss is the classification loss minus the transfer loss
(Eq. 16 with lambda = 1). Adam, learning rate 0.001, weight decay 0.001,
30 epochs. A batch of a single sample is skipped (batch normalisation cannot
train on it).

Evaluation (the reference's main.py, test2): evaluation mode with the
dropout layers switched back on (MC Dropout), the mean of 10 Monte Carlo
predictions. The reference evaluates once, after the last epoch. Here the
test samples are also scored after every epoch, with the random number
generators saved and restored around it, so that training is the same as
without these evaluations.

  final_epoch         after the last epoch: the reference's score and version 27's protocol (no test label used)
  highest_test_epoch  the epoch with the highest test accuracy, chosen with the test labels; for comparison only
"""
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ...evaluation.complexity import synchronize
from ...evaluation.metrics import classification_metrics
from .network import build_model
from .settings import ModelSettings, TrainingSettings

SCORES = ("final_epoch", "highest_test_epoch")


def enable_dropout(model) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model, x, device):
    """Mean probabilities of the Monte Carlo passes, without changing the training's random numbers."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        model.eval()
        enable_dropout(model)
        probabilities = model.predict(x)
    model.train()
    return probabilities


def run_fold(source, target, device, training: TrainingSettings = TrainingSettings(),
             model_settings: ModelSettings = ModelSettings(), description: str = "", show_progress: bool = True):
    """`source`: (features, labels) of the training samples; `target`: (features, labels) of the test samples
    (the labels only for scoring). Returns the fold's record and the final state."""
    torch.manual_seed(training.seed)
    x, y = (torch.as_tensor(a, device=device) for a in source)
    tx, ty = (torch.as_tensor(a, device=device) for a in target)
    model = build_model(x.shape[1], model_settings, training).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    target_batches = [tx[i:i + training.target_batch] for i in range(0, len(tx), training.target_batch)]
    target_batches = [b for b in target_batches if len(b) > 1] or [tx]
    labels = ty.tolist()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    history = {key: [] for key in ("classification_loss", "transfer_loss", "train_accuracy", "test_loss",
                                   "test_accuracy")}
    test_probabilities, step = [], 0
    for _ in tqdm(range(training.epochs), desc=description, leave=False, disable=not show_progress):
        model.train()
        order = torch.randperm(len(x), device=device)
        batches = [order[i:i + training.source_batch] for i in range(0, len(order), training.source_batch)]
        batches = [b for b in batches if len(b) > 1]
        model.adapt_loss.update_dynamic(len(batches))
        totals = torch.zeros(3, device=device)
        for batch in batches:
            target_x = target_batches[step % len(target_batches)]
            step += 1
            classification, transfer = model(x[batch], target_x, y[batch])
            loss = (classification - transfer).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += torch.stack([classification.detach().sum(), transfer.detach(), torch.zeros((), device=device)])
        probabilities = mc_predict(model, tx, device)
        test_probabilities.append(probabilities.cpu())
        source_accuracy = (mc_predict(model, x, device).argmax(1) == y).float().mean()
        for key, value in (("classification_loss", totals[0] / len(batches)), ("transfer_loss", totals[1] / len(batches)),
                           ("train_accuracy", source_accuracy),
                           ("test_loss", F.nll_loss(probabilities.clamp(min=1e-12).log(), ty)),
                           ("test_accuracy", (probabilities.argmax(1) == ty).float().mean())):
            history[key].append(round(float(value), 5))
    synchronize(device)
    seconds = time.perf_counter() - start

    highest = int(np.argmax(history["test_accuracy"]))
    record: dict[str, Any] = dict(labels=labels)
    for name, epoch in (("final_epoch", training.epochs - 1), ("highest_test_epoch", highest)):
        p = test_probabilities[epoch]
        predictions, scores = p.argmax(1).tolist(), p[:, 1].tolist()
        record[name] = dict(classification_metrics(labels, predictions, scores), epoch=epoch + 1)
        record[f"{name}_predictions"], record[f"{name}_scores"] = predictions, scores
    record.update({f"{key}_per_epoch": values for key, values in history.items()},
                  dynamic_factor_final=float(model.adapt_loss.dynamic_factor), training_seconds=seconds,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  training_samples=len(source[1]), testing_samples=len(labels))
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return record, state
