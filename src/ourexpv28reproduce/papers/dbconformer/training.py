"""DBConformer training of one fold, as DBConformer_LOSO.py and DBConformer_CV.py do it.

Adam (learning rate 0.001), cross-entropy, shuffled batches of 32, 100
epochs; a batch of a single sample is skipped, and leave one subject out
drops the last incomplete batch (the code's loaders). The test trials are
scored after every epoch. The code reports the accuracy after the last epoch;
nothing is selected.

  final_epoch         after the last epoch, no test label used: the paper's score and version 27's protocol
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
EVALUATION_BATCH = 96          # the code scores in batches of 3 x 32


@torch.no_grad()
def evaluate(model, x):
    model.eval()
    return torch.cat([model(x[i:i + EVALUATION_BATCH]) for i in range(0, len(x), EVALUATION_BATCH)])


def run_fold(train, test, depth: int, device, drop_last: bool, training: TrainingSettings = TrainingSettings(),
             model_settings: ModelSettings = ModelSettings(), description: str = "", show_progress: bool = True):
    """`train`, `test`: (x [n, channels, time] aligned, labels). Returns the fold's record and the final state."""
    torch.manual_seed(training.seed)
    x, y = (torch.as_tensor(a, device=device) for a in (np.asarray(train[0], np.float32), train[1]))
    tx, ty = (torch.as_tensor(a, device=device) for a in (np.asarray(test[0], np.float32), test[1]))
    model = build_model(x.shape[1], x.shape[2], depth, settings=model_settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    labels = ty.tolist()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    curves, test_logits = [], []
    for _ in tqdm(range(training.epochs), desc=description, leave=False, disable=not show_progress):
        model.train()
        order = torch.randperm(len(x), device=device)
        stop = len(order) - len(order) % training.batch_size if drop_last else len(order)
        total, correct, count = (torch.zeros((), device=device) for _ in range(3))
        for i in range(0, stop, training.batch_size):
            batch = order[i:min(i + training.batch_size, stop)]
            if len(batch) == 1:
                continue
            logits = model(x[batch])
            loss = F.cross_entropy(logits, y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(batch)
            correct += (logits.argmax(dim=1) == y[batch]).sum()
            count += len(batch)
        logits = evaluate(model, tx)
        test_logits.append(logits)
        curves.append(torch.stack([total / count, correct / count, F.cross_entropy(logits, ty),
                                   (logits.argmax(dim=1) == ty).float().mean()]))
    synchronize(device)
    seconds = time.perf_counter() - start

    curves = torch.stack(curves).T.tolist()
    test_logits = torch.stack(test_logits)
    highest = int(np.argmax(curves[3]))
    record: dict[str, Any] = dict(labels=labels)
    for name, epoch in (("final_epoch", training.epochs - 1), ("highest_test_epoch", highest)):
        predictions = test_logits[epoch].argmax(dim=1).tolist()
        scores = test_logits[epoch].softmax(dim=1)[:, 1].tolist()
        record[name] = dict(classification_metrics(labels, predictions, scores), epoch=epoch + 1)
        record[f"{name}_predictions"], record[f"{name}_scores"] = predictions, scores
    for key, values in zip(("train_loss", "train_accuracy", "test_loss", "test_accuracy"), curves):
        record[f"{key}_per_epoch"] = [round(v, 5) for v in values]
    record.update(training_seconds=seconds,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  training_samples=len(train[1]), testing_samples=len(labels))
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return record, state
