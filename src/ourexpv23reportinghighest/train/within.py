"""Within-subject cross-validation: train and test on trials of the same subject.

Every subject is evaluated alone. Its trials are split into stratified folds;
for each fold the ensemble is trained on the other folds, as in version 23,
and the fold is scored after every epoch. Two results are kept for a fold:
the final epoch, and the epoch with the highest test accuracy, which is
chosen with the test labels (see engine.score_epochs).

Two differences from the leave-one-subject-out run:
  - Logits are not centred. A test fold has about 14 trials of a subject the
    model was trained on, so there is no subject-wide bias to remove.
  - Per-subject alignment is computed from all trials of the subject,
    including the test fold's, without their labels.
"""
from dataclasses import asdict, replace
from importlib.metadata import version

import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

from ..data import DataConfig, build_subject_graphs
from ..model import ModelConfig
from .config import TrainConfig
from .loso import EVALUATION, print_scores, run_fold, select_device
from .metrics import SCORES, classification_metrics, summarize_both

# Training settings that differ from the leave-one-subject-out defaults.
# A fold has about 130 training trials (five batches per epoch), so more
# epochs are needed; the value was chosen on validation splits taken from
# training folds only.
WITHIN_DEFAULTS = dict(epochs=60, center_logits=False)


def within_config(**overrides) -> TrainConfig:
    return replace(TrainConfig(), **{**WITHIN_DEFAULTS, **overrides})


def fold_indices(labels, folds: int, split_seed: int):
    """Stratified, shuffled folds: [(training indices, testing indices), ...]."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    return list(splitter.split(np.zeros(len(labels)), labels))


def pooled_metrics(fold_results, score: str) -> dict:
    """Scores of a subject over all its trials, each predicted by the fold that did not train on it."""
    confusion = np.sum([fold[score]["confusion_matrix"] for fold in fold_results], axis=0)
    true = [0] * int(confusion[0].sum()) + [1] * int(confusion[1].sum())
    predicted = ([0] * int(confusion[0, 0]) + [1] * int(confusion[0, 1])
                 + [0] * int(confusion[1, 0]) + [1] * int(confusion[1, 1]))
    accuracies = [fold[score]["accuracy"] for fold in fold_results]
    return dict(classification_metrics(true, predicted), fold_accuracy_min=float(min(accuracies)),
                fold_accuracy_max=float(max(accuracies)))


def run_subject(graphs, subject: int, model_config: ModelConfig, config: TrainConfig, device,
                folds: int, split_seed: int) -> dict:
    """Cross-validate one subject. Every trial is tested exactly once."""
    labels = [int(graph.y) for graph in graphs]
    fold_results = []
    for number, (train_index, test_index) in enumerate(fold_indices(labels, folds, split_seed), start=1):
        result = run_fold([graphs[i] for i in train_index], [graphs[i] for i in test_index],
                          f"S{subject} fold {number}/{folds}", model_config, config, device)
        fold_results.append(dict(fold=number, **result))
    return dict({score: pooled_metrics(fold_results, score) for score in SCORES}, folds=fold_results)


def run_within(data_dir, data_config: DataConfig = DataConfig(), model_config: ModelConfig = ModelConfig(),
               config: TrainConfig | None = None, subjects=None, folds: int = 10, split_seed: int = 42) -> dict:
    """Cross-validate every subject in `subjects` (default: all) separately."""
    config = config or within_config()
    config.validate()
    device = select_device(config.device)
    subjects = tuple(subjects or data_config.subjects)
    if not set(subjects) <= set(data_config.subjects):
        raise ValueError("Subjects must be among the configured subjects")
    results = {}
    for subject in subjects:
        print(f"S{subject}: building graphs", flush=True)
        graphs = build_subject_graphs(data_dir, subject, data_config)
        results[str(subject)] = run_subject(graphs, subject, model_config, config, device, folds, split_seed)
        print_scores(f"S{subject} ({len(graphs)} trials)", results[str(subject)])
    summary = summarize_both(results)
    print_scores("Mean over subjects", summary)
    return dict(protocol="within_subject_cross_validation", evaluation=EVALUATION,
                shuffled_training_labels=config.shuffle_training_labels,
                folds=folds, split_seed=split_seed, summary=summary, subjects=results,
                data_config=asdict(data_config), model_config=asdict(model_config), train_config=asdict(config),
                unlabeled_test_trial_use=dict(
                    alignment="Euclidean alignment uses all trials of the subject, including each test fold's",
                    center_logits=config.center_logits,
                    note="No labels of test trials are used."),
                device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                packages={name: version(name) for name in ("torch", "torch-geometric", "numpy", "scipy", "scikit-learn")})
