"""EEGNet training of one fold, as in the paper (section 2.2.1) and the code's example (examples/ERP.py).

Adam with Keras defaults, categorical cross-entropy, shuffled mini-batches,
500 epochs, and "validation stopping, saving the model weights which produced
the lowest validation set loss". The validation set comes from the training
side of the fold (other subjects, or another fold of the same subject), so
the selected model is chosen without the test labels. A class weight (inverse
proportion, majority class 1, rounded up) is applied only when the training
classes are unequal, as in section 2.3; our samples are balanced.

Three scores are kept from the same run:

  validation_selected  the weights with the lowest validation loss: the paper's protocol, no test label used
  final_epoch          after the last epoch: our protocol for version 27, no test label used
  highest_test_epoch   the epoch with the highest test accuracy, chosen with the test labels: biased upward,
                       recorded only to compare with work that reports this way

Recorded per epoch: loss and accuracy on the training batches (training
mode), on the validation set and on the test samples; also the training time
and peak GPU memory.
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


def class_weights(labels, classes: int = 2):
    """Section 2.3: inverse proportion in the training data, majority class 1, rounded up; None if balanced."""
    counts = np.bincount(labels, minlength=classes)
    if len(set(counts.tolist())) == 1:
        return None
    return torch.tensor([math.ceil(counts.max() / max(c, 1)) for c in counts], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, x, y):
    """(loss, accuracy, predictions, probability of class 1) in evaluation mode."""
    model.eval()
    logits = torch.cat([model(x[i:i + EVALUATION_BATCH]) for i in range(0, len(x), EVALUATION_BATCH)])
    probabilities = logits.softmax(dim=1)
    predictions = logits.argmax(dim=1)
    return (float(F.cross_entropy(logits, y)), float((predictions == y).float().mean()), predictions.tolist(),
            probabilities[:, 1].tolist())


def run_fold(train, validation, test, dropout: float, device, training: TrainingSettings = TrainingSettings(),
             model_settings: ModelSettings = ModelSettings(), description: str = "", show_progress: bool = True):
    """`train`, `validation`, `test`: (x [n, channels, time], labels). Returns the fold's record and the
    state of the validation-selected model."""
    torch.manual_seed(training.seed)
    generator = torch.Generator().manual_seed(training.seed)
    tensors = [(torch.as_tensor(x, device=device), torch.as_tensor(y, device=device)) for x, y in (train, validation,
                                                                                                    test)]
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = tensors
    model = build_model(x_train.shape[1], x_train.shape[2], dropout, settings=model_settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate, eps=training.adam_epsilon)
    weight = class_weights(train[1])
    weight = None if weight is None else weight.to(device)
    labels = y_test.tolist()

    history = {key: [] for key in ("train_loss", "train_accuracy", "validation_loss", "validation_accuracy",
                                   "test_loss", "test_accuracy")}
    best_val, best_test_accuracy, selected, highest, selected_state = math.inf, -1.0, None, None, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    for epoch in tqdm(range(1, training.epochs + 1), desc=description, leave=False, disable=not show_progress):
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(device)
        total, correct = 0.0, 0
        for i in range(0, len(order), training.batch_size):
            batch = order[i:i + training.batch_size]
            logits = model(x_train[batch])
            loss = F.cross_entropy(logits, y_train[batch], weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.apply_constraints()
            total += loss.item() * len(batch)
            correct += int((logits.argmax(dim=1) == y_train[batch]).sum())
        val_loss, val_accuracy, _, _ = evaluate(model, x_val, y_val)
        test_loss, test_accuracy, predictions, scores = evaluate(model, x_test, y_test)
        for key, value in (("train_loss", total / len(order)), ("train_accuracy", correct / len(order)),
                           ("validation_loss", val_loss), ("validation_accuracy", val_accuracy),
                           ("test_loss", test_loss), ("test_accuracy", test_accuracy)):
            history[key].append(value)
        if val_loss < best_val:                     # ModelCheckpoint(save_best_only=True) on val_loss
            best_val, selected = val_loss, (epoch, predictions, scores)
            selected_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if test_accuracy > best_test_accuracy:      # the first epoch with the highest test accuracy
            best_test_accuracy, highest = test_accuracy, (epoch, predictions, scores)
    synchronize(device)
    seconds = time.perf_counter() - start

    record: dict[str, Any] = dict(labels=labels)
    for name, (epoch, predictions, scores) in zip(SCORES, (selected, (training.epochs, predictions, scores), highest)):
        record[name] = dict(classification_metrics(labels, predictions, scores), epoch=epoch)
        record[f"{name}_predictions"], record[f"{name}_scores"] = predictions, scores
    record.update({f"{key}_per_epoch": values for key, values in history.items()},
                  training_seconds=seconds,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  training_samples=len(train[1]), validation_samples=len(validation[1]), testing_samples=len(labels))
    return record, selected_state
