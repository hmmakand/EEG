"""Validation-only epoch selection, with independent adaptation for each target subject."""
from dataclasses import dataclass
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..model import EEGGraphModel
from .calibration import TargetCalibration, train_epoch_calibrated
from .complexity import synchronize
from .engine import ensemble_scores, test_logits, train_epoch
from .metrics import classification_metrics


def split_subjects(subjects, held_out, seed, fraction=0.2):
    """Deterministic subject split; ceil gives 9/2/3 validation subjects for Liu/BCI/Jia."""
    others = sorted(s for s in subjects if s != held_out)
    if held_out not in subjects or len(others) < 2:
        raise ValueError("LOSO validation needs at least three subjects including the held-out subject")
    count = min(len(others) - 1, max(1, int(np.ceil(fraction * len(others)))))
    rng = np.random.default_rng([seed, held_out])
    validation = sorted(int(s) for s in rng.choice(others, count, replace=False))
    return [s for s in others if s not in validation], validation


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best_score: float = -float('inf')
    best_epoch: int = 0
    stale: int = 0

    def update(self, epoch, score):
        if not np.isfinite(score):
            raise ValueError("Non-finite validation balanced accuracy")
        if score > self.best_score + self.min_delta:
            self.best_score, self.best_epoch, self.stale = float(score), epoch, 0
        else:
            self.stale += 1
        return self.stale >= self.patience


class Member:
    """One model/optimizer with its own RNG stream, allowing lockstep ensemble selection."""

    def __init__(self, training, target, model_config, config, seed, device):
        self.device, self.config = device, config
        self.devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        self.loader = DataLoader(training, batch_size=config.batch_size, shuffle=True)
        self.batches = list(DataLoader(target, batch_size=config.batch_size, shuffle=False))
        self.calibration = TargetCalibration(training, target, config) if config.target_calibration else None
        with torch.random.fork_rng(devices=self.devices):
            torch.manual_seed(seed)
            self.model = EEGGraphModel(model_config).to(device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate,
                                              weight_decay=config.weight_decay)
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(device) if self.devices else None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.history = dict(train_loss=[], train_accuracy=[])

    def epoch(self, number, score_target=True):
        with torch.random.fork_rng(devices=self.devices):
            torch.set_rng_state(self.cpu_rng)
            if self.devices:
                torch.cuda.set_rng_state(self.cuda_rng, self.device)
            if self.calibration is not None and number > self.config.calibration_warmup_epochs:
                loss, accuracy = train_epoch_calibrated(self.model, self.loader, self.calibration,
                                                        self.optimizer, self.criterion, self.device)
            else:
                loss, accuracy = train_epoch(self.model, self.loader, self.optimizer, self.criterion, self.device)
            self.history['train_loss'].append(loss)
            self.history['train_accuracy'].append(accuracy)
            # In refit, intermediate target forwards serve only label-free calibration.
            needs_calibration = self.calibration is not None and number >= self.config.calibration_warmup_epochs
            logits = test_logits(self.model, self.batches, self.device) if score_target or needs_calibration else None
            if needs_calibration:
                self.calibration.refresh(logits)
            self.cpu_rng = torch.get_rng_state()
            if self.devices:
                self.cuda_rng = torch.cuda.get_rng_state(self.device)
        return logits


def select_epochs(training, validation_groups, model_config, config, device, description='Validation'):
    """No outer-test argument exists here. Each validation target adapts separately.

    Without calibration the source ensemble is shared; scoring/logit centering is
    still separate by subject. With calibration there is one independent source
    ensemble per validation subject, matching single-target LOSO deployment.
    """
    if not training or not validation_groups or any(not g for g in validation_groups.values()):
        raise ValueError('Training and validation partitions must be non-empty')
    for graphs in [training, *validation_groups.values()]:
        if {int(g.y) for g in graphs} != {0, 1}:
            raise ValueError('Training and each validation group must contain both classes')
    synchronize(device)
    start = time.perf_counter()
    first = next(iter(validation_groups))
    keys = list(validation_groups) if config.target_calibration else [first]
    ensembles = {key: [Member(training, validation_groups[key], model_config, config,
                             config.seed + m, device) for m in range(config.ensemble)] for key in keys}
    batches = {key: list(DataLoader(graphs, batch_size=config.batch_size, shuffle=False))
               for key, graphs in validation_groups.items()}
    labels = {key: [int(g.y) for g in graphs] for key, graphs in validation_groups.items()}
    stopper = EarlyStopping(config.patience, config.min_delta)
    history = []
    for epoch in tqdm(range(1, config.epochs + 1), desc=description, disable=not config.show_progress):
        current = {key: torch.stack([member.epoch(epoch) for member in ensemble])
                   for key, ensemble in ensembles.items()}
        scores = {}
        for key in validation_groups:
            logits = current[key] if key in current else torch.stack([
                test_logits(member.model, batches[key], device) for member in ensembles[first]])
            predictions, probability = ensemble_scores(logits, config.center_logits)
            scores[str(key)] = classification_metrics(labels[key], predictions.tolist(), probability.tolist())['balanced_accuracy']
        mean = float(np.mean(list(scores.values())))
        history.append(dict(epoch=epoch, balanced_accuracy=mean, by_group=scores))
        if stopper.update(epoch, mean):
            break
    synchronize(device)
    return dict(selected_epoch=stopper.best_epoch, best_balanced_accuracy=stopper.best_score,
                stopped_epoch=epoch, early_stopped=epoch < config.epochs,
                stop_reason='patience' if stopper.stale >= config.patience else 'max_epochs',
                max_epochs=config.epochs, patience=config.patience, min_delta=config.min_delta,
                metric='mean_validation_balanced_accuracy', history=history,
                independent_target_ensembles=config.target_calibration,
                training_samples=len(training), validation_samples={str(k): len(v) for k, v in validation_groups.items()},
                training_seconds=time.perf_counter() - start)
