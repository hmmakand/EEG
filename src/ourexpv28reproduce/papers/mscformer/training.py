"""MSCFormer training of one test fold: five models, one per validation part (paper and MSCFormer.ipynb).

The training side of the fold is z-scored with its own mean and standard
deviation (the test samples with the same values) and split five ways; model
m trains on four parts with the fifth as its validation set, for 1,000 epochs
with Adam (learning rate 0.001, betas 0.5 and 0.999), cross-entropy, and
batches of 72 real samples plus 216 S&R samples (three times as many, eight
segments) generated from the model's own training parts.

"We selected the model with the lowest loss on the validation set as the best
model and tested it on the test set. This process was repeated for each of
the five original training subsets, and the final performance metric was
obtained by averaging the results on the test set." The five models are
therefore scored separately and their metrics averaged (experiment.py); they
do not vote.

Three scores per model, as for the other papers:

  validation_selected  the epoch with the lowest validation loss: the paper's protocol
  final_epoch          after the last epoch: version 27's protocol
  highest_test_epoch   the epoch with the highest test accuracy, chosen with the test labels; for comparison only
"""
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ...datascripts.augmentation import segment_and_reconstruct
from ...evaluation.complexity import synchronize
from .network import build_model
from .settings import ModelSettings, TrainingSettings

SCORES = ("validation_selected", "final_epoch", "highest_test_epoch")
EVALUATION_BATCH = 256


@torch.no_grad()
def probabilities(model, x):
    model.eval()
    return torch.cat([model(x[i:i + EVALUATION_BATCH]).softmax(dim=1) for i in range(0, len(x), EVALUATION_BATCH)])


def train_member(train, validation, x_test, y_test, branch_dropout: float, seed: int, device,
                 training: TrainingSettings, model_settings: ModelSettings, description: str, show_progress: bool):
    """One model. Returns {score: (epoch, test probabilities [n, classes])}, its curves and the state of its
    validation-selected epoch."""
    torch.manual_seed(seed)
    (x, y), (x_val, y_val) = [(torch.as_tensor(a, device=device), torch.as_tensor(b, device=device))
                              for a, b in (train, validation)]
    model = build_model(x.shape[1], x.shape[2], branch_dropout=branch_dropout, settings=model_settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate, betas=training.betas,
                                 weight_decay=training.weight_decay)
    extra = training.augmentation_factor * training.batch_size
    curves, best_loss, best_test, chosen, best_state = [], math.inf, -1.0, {}, None
    for epoch in tqdm(range(1, training.epochs + 1), desc=description, leave=False, disable=not show_progress):
        model.train()
        order = torch.randperm(len(x), device=device)
        total, correct, count = (torch.zeros((), device=device) for _ in range(3))
        for i in range(0, len(order), training.batch_size):
            batch = order[i:i + training.batch_size]
            extra_x, extra_y = segment_and_reconstruct(x, y, extra, training.segments)
            inputs, targets = torch.cat([x[batch], extra_x]), torch.cat([y[batch], extra_y])
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(targets)
            correct += (logits.argmax(dim=1) == targets).sum()
            count += len(targets)
        val, test = probabilities(model, x_val), probabilities(model, x_test)
        val_loss = F.nll_loss(val.clamp(min=1e-12).log(), y_val)
        test_accuracy = (test.argmax(1) == y_test).float().mean()
        curves.append(torch.stack([total / count, correct / count, val_loss, (val.argmax(1) == y_val).float().mean(),
                                   F.nll_loss(test.clamp(min=1e-12).log(), y_test), test_accuracy]))
        loss_value, accuracy_value = (float(v) for v in (val_loss, test_accuracy))
        if loss_value < best_loss:
            best_loss = loss_value
            chosen["validation_selected"] = (epoch, test)
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if accuracy_value > best_test:
            best_test = accuracy_value
            chosen["highest_test_epoch"] = (epoch, test)
    chosen["final_epoch"] = (training.epochs, test)
    keys = ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "test_loss", "test_accuracy")
    history = dict(zip(keys, torch.stack(curves).T.tolist()))
    return {name: (epoch, probs.cpu()) for name, (epoch, probs) in chosen.items()}, history, best_state


def run_fold(train, x_test, y_test, splits, branch_dropout: float, device,
             training: TrainingSettings = TrainingSettings(), model_settings: ModelSettings = ModelSettings(),
             description: str = "", show_progress: bool = True):
    """`train`: (x, y) of the fold's training side, already z-scored; `splits`: one (fit indices, validation
    indices) into it per model. Returns the fold's record (per model: epochs and test probabilities) and the
    validation-selected state of every model."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    x, y = train
    test_x, test_y = torch.as_tensor(x_test, device=device), torch.as_tensor(y_test, device=device)
    synchronize(device)
    start = time.perf_counter()
    members, histories, states = [], [], []
    for m, (fit, val) in enumerate(splits):
        chosen, history, state = train_member((x[fit], y[fit]), (x[val], y[val]), test_x, test_y, branch_dropout,
                                              training.seed + m, device, training, model_settings,
                                              f"{description}, model {m + 1}/{len(splits)}", show_progress)
        members.append(dict(training_samples=len(fit), validation_samples=len(val),
                            **{f"{name}_epoch": epoch for name, (epoch, _) in chosen.items()},
                            **{f"{name}_probabilities": probs[:, 1].tolist() for name, (_, probs) in chosen.items()}))
        histories.append(history)
        states.append(state)
    synchronize(device)
    record: dict[str, Any] = dict(labels=test_y.tolist(), members=members)
    for key in histories[0]:
        record[f"{key}_per_epoch"] = [round(v, 5) for v in np.mean([h[key] for h in histories], axis=0).tolist()]
    record.update(training_seconds=time.perf_counter() - start,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  testing_samples=len(y_test))
    return record, states
