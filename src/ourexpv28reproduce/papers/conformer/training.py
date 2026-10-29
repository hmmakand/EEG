"""EEG Conformer training of one fold, as conformer.py does it.

Adam (learning rate 0.0002, betas 0.5 and 0.999), cross-entropy, batches of
72 shuffled training samples, 2,000 epochs. Every batch is extended with as
many S&R samples (segmentation and reconstruction, section III-B): for each
class, batch / classes new samples, each made of Ns = 8 consecutive time
segments taken from randomly chosen training samples of that class, in their
original time order. The test samples are scored after every epoch.

The code has no validation set and reports "the best accuracy", the highest
test accuracy over the 2,000 epochs. Two scores are kept from the same run:

  final_epoch         after the last epoch; no test label is used (version 27's protocol)
  highest_test_epoch  the code's reported score: the epoch with the highest test accuracy, chosen with the
                      test labels; biased upward, for comparison with the paper
The code's other printed number, the test accuracy averaged over all epochs,
is recorded per fold as `mean_test_accuracy_over_epochs`.
"""
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ...datascripts.augmentation import segment_and_reconstruct
from ...evaluation.complexity import synchronize
from ...evaluation.metrics import classification_metrics
from .network import build_model
from .preparation import z_score
from .settings import ModelSettings, TrainingSettings

SCORES = ("final_epoch", "highest_test_epoch")
EVALUATION_BATCH = 256


@torch.no_grad()
def evaluate(model, x):
    model.eval()
    return torch.cat([model(x[i:i + EVALUATION_BATCH]) for i in range(0, len(x), EVALUATION_BATCH)])


def run_fold(train, test, device, training: TrainingSettings = TrainingSettings(),
             model_settings: ModelSettings = ModelSettings(), description: str = "", show_progress: bool = True):
    """`train`, `test`: (x [n, channels, time] filtered, labels). Returns the fold's record and the final state."""
    torch.manual_seed(training.seed)
    x_train, x_test = z_score(train[0], test[0])
    x, y = torch.as_tensor(x_train, device=device), torch.as_tensor(train[1], device=device)
    tx, ty = torch.as_tensor(x_test, device=device), torch.as_tensor(test[1], device=device)
    model = build_model(x.shape[1], x.shape[2], settings=model_settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate, betas=training.betas)
    labels = ty.tolist()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    curves, test_logits = [], []
    for _ in tqdm(range(training.epochs), desc=description, leave=False, disable=not show_progress):
        model.train()
        order = torch.randperm(len(x), device=device)
        total, correct, count = (torch.zeros((), device=device) for _ in range(3))
        for i in range(0, len(order), training.batch_size):
            batch = order[i:i + training.batch_size]
            extra_x, extra_y = segment_and_reconstruct(x, y, training.batch_size, training.segments)
            inputs, targets = torch.cat([x[batch], extra_x]), torch.cat([y[batch], extra_y])
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(targets)
            correct += (logits.argmax(dim=1) == targets).sum()
            count += len(targets)
        logits = evaluate(model, tx)
        test_logits.append(logits)
        curves.append(torch.stack([total / count, correct / count, F.cross_entropy(logits, ty),
                                   (logits.argmax(dim=1) == ty).float().mean()]))
    synchronize(device)
    seconds = time.perf_counter() - start

    curves = torch.stack(curves).T.tolist()
    test_logits = torch.stack(test_logits)
    accuracies = curves[3]
    highest = int(np.argmax(accuracies))                     # the first epoch with the highest test accuracy
    record: dict[str, Any] = dict(labels=labels)
    for name, epoch in (("final_epoch", training.epochs - 1), ("highest_test_epoch", highest)):
        predictions = test_logits[epoch].argmax(dim=1).tolist()
        scores = test_logits[epoch].softmax(dim=1)[:, 1].tolist()
        record[name] = dict(classification_metrics(labels, predictions, scores), epoch=epoch + 1)
        record[f"{name}_predictions"], record[f"{name}_scores"] = predictions, scores
    for key, values in zip(("train_loss", "train_accuracy", "test_loss", "test_accuracy"), curves):
        record[f"{key}_per_epoch"] = [round(v, 5) for v in values]
    record.update(mean_test_accuracy_over_epochs=float(np.mean(accuracies)), training_seconds=seconds,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  training_samples=len(train[1]), testing_samples=len(labels))
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return record, state
