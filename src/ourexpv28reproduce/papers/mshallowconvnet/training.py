"""M-ShallowConvNet training of one test fold: five models, bagged (section II-E and the published code).

The training side of the fold is split five ways; model m is trained on four
parts with the fifth as its validation set, for 1,000 epochs with Adam
(learning rate 0.002, weight decay 0.075), cosine annealing to 0, batch 64,
and CutCat augmentation. Each model keeps its checkpoint with the lowest
validation loss ("selected the checkpoint with the lowest loss value as the
final model"). The five models vote on the test samples, as evaluation.py
does (one-hot predictions averaged); the probability used for the ROC curve
is the mean of the five models' probabilities.

CutCat (section II-C): every training sample X1 gets a partner X2 of the other
class from the same mini-batch; a stretch of X1 centred at c ~ U(0, T) and
r * T long, r ~ U(0, lambda), is replaced by X2's, and the labels are mixed
as (1 - r) y1 + r y2, with r the share actually replaced. A sample with no
other-class sample in its batch is left unchanged.

Three scores are kept from the same run, as for the other papers:

  validation_selected  every model's lowest-validation-loss checkpoint, bagged: the paper's protocol
  final_epoch          every model after the last epoch, bagged: version 27's protocol
  highest_test_epoch   the epoch at which the bagged models are most accurate on the test samples,
                       chosen with the test labels: for comparison only
"""
import math
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

SCORES = ("validation_selected", "final_epoch", "highest_test_epoch")
EVALUATION_BATCH = 256


def cutcat(x, y, ratio: float, classes: int = 2):
    """CutCat on one batch: (mixed x, soft labels [batch, classes])."""
    batch, _, length = x.shape
    labels = F.one_hot(y, classes).float()
    other = y.unsqueeze(0) != y.unsqueeze(1)                                   # [i, j]: j has another class
    has_partner = other.any(dim=1)
    partner = torch.where(other, torch.rand(batch, batch, device=x.device), torch.full_like(other, -1.0,
                                                                                           dtype=torch.float))
    partner = partner.argmax(dim=1)
    # As augmentation.py: a length up to ratio * T, a centre anywhere, clipped to the sample.
    span = (torch.rand(batch, device=x.device) * int(length * ratio)).long()
    centre = (torch.rand(batch, device=x.device) * length).long()
    start = (centre - span // 2).clamp(0, length)
    stop = (centre + span // 2).clamp(0, length)
    time_index = torch.arange(length, device=x.device)
    replaced = (time_index >= start[:, None]) & (time_index < stop[:, None]) & has_partner[:, None]
    mixed = torch.where(replaced[:, None, :], x[partner], x)
    share = replaced.float().mean(dim=1, keepdim=True)
    return mixed, (1 - share) * labels + share * labels[partner]


@torch.no_grad()
def probabilities(model, x):
    model.eval()
    return torch.cat([model(x[i:i + EVALUATION_BATCH]).softmax(dim=1) for i in range(0, len(x), EVALUATION_BATCH)])


def train_member(train, validation, x_test, y_test, ratio: float, seed: int, device,
                 training: TrainingSettings, model_settings: ModelSettings, description: str, show_progress: bool):
    """One model. Returns its test probabilities after every epoch [epochs, n, classes], the epoch of its lowest
    validation loss, and its curves."""
    torch.manual_seed(seed)
    (x, y), (x_val, y_val) = [(torch.as_tensor(a, device=device), torch.as_tensor(b, device=device))
                              for a, b in (train, validation)]
    model = build_model(x.shape[1], x.shape[2], settings=model_settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training.epochs, eta_min=0)
    history = {key: [] for key in ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy",
                                   "test_loss", "test_accuracy")}
    test_probabilities, best_loss, best_epoch, best_state = [], math.inf, None, None
    # Sums stay on the device during the epoch; one transfer per epoch (the validation loss) decides the checkpoint.
    curves = []
    for epoch in tqdm(range(1, training.epochs + 1), desc=description, leave=False, disable=not show_progress):
        model.train()
        order = torch.randperm(len(x), device=device)
        total, correct = torch.zeros((), device=device), torch.zeros((), device=device)
        for i in range(0, len(order), training.batch_size):
            batch = order[i:i + training.batch_size]
            inputs, targets = cutcat(x[batch], y[batch], ratio)
            logits = model(inputs)
            loss = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(batch)
            correct += (logits.argmax(dim=1) == targets.argmax(dim=1)).sum()
        scheduler.step()
        val = probabilities(model, x_val)
        test = probabilities(model, x_test)
        val_loss_tensor = F.nll_loss(val.clamp(min=1e-12).log(), y_val)
        curves.append(torch.stack([total / len(order), correct / len(order), val_loss_tensor,
                                   (val.argmax(1) == y_val).float().mean(),
                                   F.nll_loss(test.clamp(min=1e-12).log(), y_test),
                                   (test.argmax(1) == y_test).float().mean()]))
        test_probabilities.append(test)
        val_loss = float(val_loss_tensor)
        if val_loss < best_loss:                       # ModelCheckpoint(monitor="val_loss", mode="min")
            best_loss, best_epoch = val_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for key, values in zip(history, torch.stack(curves).T.tolist()):
        history[key] = values
    return torch.stack(test_probabilities).cpu(), best_epoch, history, best_state


def bagged(member_probabilities):
    """The vote of the models (evaluation.py: one-hot predictions averaged, then argmax) and the mean probability
    of class 1. `member_probabilities` [models, n, classes]."""
    votes = F.one_hot(member_probabilities.argmax(dim=2), member_probabilities.shape[2]).float().mean(dim=0)
    return votes.argmax(dim=1).tolist(), member_probabilities.mean(dim=0)[:, 1].tolist()


def run_fold(splits, x_test, y_test, ratio: float, device, training: TrainingSettings = TrainingSettings(),
             model_settings: ModelSettings = ModelSettings(), description: str = "", show_progress: bool = True):
    """`splits`: one ((x_train, y_train), (x_val, y_val)) per model. Returns the fold's record and the
    validation-selected state of every model."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    test_x, test_y = torch.as_tensor(x_test, device=device), torch.as_tensor(y_test, device=device)
    labels = test_y.tolist()
    synchronize(device)
    start = time.perf_counter()
    curves, epochs, states, per_epoch = [], [], [], []
    for member, (train, validation) in enumerate(splits):
        probs, best_epoch, history, state = train_member(train, validation, test_x, test_y, ratio,
                                                         training.seed + member, device, training, model_settings,
                                                         f"{description}, model {member + 1}/{len(splits)}",
                                                         show_progress)
        per_epoch.append(probs)
        epochs.append(best_epoch)
        curves.append(history)
        states.append(state)
    synchronize(device)
    seconds = time.perf_counter() - start

    per_epoch = torch.stack(per_epoch)                           # [models, epochs, n, classes]
    ensemble = [bagged(per_epoch[:, e]) for e in range(per_epoch.shape[1])]
    accuracies = [float(np.mean(np.array(p) == np.array(labels))) for p, _ in ensemble]
    highest = int(np.argmax(accuracies))
    chosen = dict(validation_selected=bagged(torch.stack([per_epoch[m, e - 1] for m, e in enumerate(epochs)])),
                  final_epoch=ensemble[-1], highest_test_epoch=ensemble[highest])
    record: dict[str, Any] = dict(labels=labels)
    for name in SCORES:
        predictions, scores = chosen[name]
        record[name] = classification_metrics(labels, predictions, scores)
        record[f"{name}_predictions"], record[f"{name}_scores"] = predictions, scores
    record["validation_selected"]["epochs"] = epochs
    record["final_epoch"]["epoch"] = training.epochs
    record["highest_test_epoch"]["epoch"] = highest + 1
    # Curves of the fold: mean over its models; the ensemble's test accuracy per epoch.
    for key in curves[0]:
        if key.endswith(("_loss", "_accuracy")):
            record[f"{key}_per_epoch"] = np.mean([c[key] for c in curves], axis=0).tolist()
    record["ensemble_test_accuracy_per_epoch"] = accuracies
    for key in [k for k in record if k.endswith("_per_epoch")]:
        record[key] = [round(value, 5) for value in record[key]]              # keeps the result files small
    record.update(models=[dict(selected_epoch=e, training_samples=len(t[1]), validation_samples=len(v[1]))
                          for e, (t, v) in zip(epochs, splits)],
                  training_seconds=seconds,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  testing_samples=len(labels))
    return record, states
