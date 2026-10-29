"""SlimSeiz training of one fold, as Loop_ADDNet_kFold.ipynb does it.

Channels: selected on the fold's training side (channels.py); the network
sees only the k selected channels.

Training: Adam (learning rate 0.001, weight decay 0.0001), batches of 32,
30 epochs; loss = 0.5 x cross-entropy + 0.5 x supervised contrastive loss
on the 32 pooled features. The code's SupervisedContrastiveLoss normalises
the features, forms their similarity matrix divided by the temperature
(0.08), and passes the rows of that matrix as embeddings to
pytorch-metric-learning's NTXentLoss (temperature 0.08); `ntxent` below is
that loss (checked against the library), so the same computation is kept.

Selection: the epoch with the highest validation accuracy (first one if
tied) is the model that is tested. Three scores per fold:

  validation_selected  the code's protocol; no test label used
  final_epoch          after the last epoch; no test label used (version 27's protocol)
  highest_test_epoch   the epoch with the highest test accuracy, chosen with the test labels; for comparison only
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
from .settings import TrainingSettings

SCORES = ("validation_selected", "final_epoch", "highest_test_epoch")
EVALUATION_BATCH = 256


def ntxent(embeddings, labels, temperature: float):
    """pytorch_metric_learning.losses.NTXentLoss with its defaults (cosine similarity, mean over positive pairs):
    for every pair (i, j) of the same class, -log(exp(s_ij) / (exp(s_ij) + sum over k of another class exp(s_ik)))."""
    e = F.normalize(embeddings, dim=1)
    s = e @ e.T / temperature
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    if not positive.any():
        return s.sum() * 0
    s = s - s.max().detach()
    negatives = (torch.exp(s) * ~same).sum(dim=1, keepdim=True)
    return -(s - torch.log(torch.exp(s) + negatives))[positive].mean()


def supervised_contrastive(features, labels, temperature: float):
    """The code's SupervisedContrastiveLoss: NTXent on the rows of the scaled cosine-similarity matrix."""
    normalized = F.normalize(features, dim=1)
    return ntxent(normalized @ normalized.T / temperature, labels, temperature)


@torch.no_grad()
def logits_of(model, x):
    model.eval()
    return torch.cat([model(x[i:i + EVALUATION_BATCH])[0] for i in range(0, len(x), EVALUATION_BATCH)])


def run_fold(train, validation, test, device, training: TrainingSettings = TrainingSettings(),
             description: str = "", show_progress: bool = True):
    """`train`, `validation`, `test`: (x [n, k selected channels, time], labels). Returns the fold's record and the
    state of the validation-selected epoch."""
    torch.manual_seed(training.seed)
    (x, y), (vx, vy), (tx, ty) = [(torch.as_tensor(a, device=device), torch.as_tensor(b, device=device))
                                  for a, b in (train, validation, test)]
    model = build_model(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    w = training.contrastive_weight
    labels = ty.tolist()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    curves, test_logits, best_val, best_epoch, best_state = [], [], -1.0, None, None
    for epoch in tqdm(range(1, training.epochs + 1), desc=description, leave=False, disable=not show_progress):
        model.train()
        order = torch.randperm(len(x), device=device)
        total, correct = torch.zeros((), device=device), torch.zeros((), device=device)
        for i in range(0, len(order), training.batch_size):
            batch = order[i:i + training.batch_size]
            logits, features = model(x[batch])
            loss = (1 - w) * F.cross_entropy(logits, y[batch]) + w * supervised_contrastive(
                features, y[batch], training.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(batch)
            correct += (logits.argmax(dim=1) == y[batch]).sum()
        val_logits, logits = logits_of(model, vx), logits_of(model, tx)
        test_logits.append(logits)
        val_accuracy = (val_logits.argmax(1) == vy).float().mean()
        curves.append(torch.stack([total / len(order), correct / len(order), F.cross_entropy(val_logits, vy),
                                   val_accuracy, F.cross_entropy(logits, ty), (logits.argmax(1) == ty).float().mean()]))
        if float(val_accuracy) > best_val:                     # "if test_accuracy > best_test_accuracy" on validation
            best_val, best_epoch = float(val_accuracy), epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    synchronize(device)
    seconds = time.perf_counter() - start

    keys = ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy", "test_loss", "test_accuracy")
    history = dict(zip(keys, torch.stack(curves).T.tolist()))
    test_logits = torch.stack(test_logits)
    highest = int(np.argmax(history["test_accuracy"])) + 1
    record: dict[str, Any] = dict(labels=labels)
    for name, epoch in (("validation_selected", best_epoch), ("final_epoch", training.epochs),
                        ("highest_test_epoch", highest)):
        predictions = test_logits[epoch - 1].argmax(dim=1).tolist()
        scores = test_logits[epoch - 1].softmax(dim=1)[:, 1].tolist()
        record[name] = dict(classification_metrics(labels, predictions, scores), epoch=epoch)
        record[f"{name}_predictions"], record[f"{name}_scores"] = predictions, scores
    record.update({f"{key}_per_epoch": [round(v, 5) for v in values] for key, values in history.items()},
                  training_seconds=seconds,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  training_samples=len(train[1]), validation_samples=len(validation[1]), testing_samples=len(labels))
    return record, best_state
