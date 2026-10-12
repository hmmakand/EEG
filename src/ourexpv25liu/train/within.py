"""Within-subject cross-validation: train and test on trials of the same subject.

Every subject is evaluated alone. Its trials are split into stratified folds;
for each fold the ensemble is trained on the other folds and the fold is
scored once, after the final epoch. Nothing is selected on test trials: the
number of epochs is fixed in advance (see WITHIN_DEFAULTS).

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
from .loso import PACKAGES, run_fold, select_device
from .metrics import classification_metrics, summarize

# Training settings that differ from the leave-one-subject-out defaults.
# They are those of version 23, where a fold of BCI IV 2a has about 130
# training trials; 60 epochs was chosen there on validation splits taken from
# training folds. A liu2024 fold has 36 training trials; the value was not
# re-tuned for it.
WITHIN_DEFAULTS = dict(epochs=60, center_logits=False)


def within_config(**overrides) -> TrainConfig:
    return replace(TrainConfig(), **{**WITHIN_DEFAULTS, **overrides})


def fold_indices(labels, folds: int, split_seed: int):
    """Stratified, shuffled folds: [(training indices, testing indices), ...]."""
    if folds < 2:
        raise ValueError("At least two folds are required")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    return list(splitter.split(np.zeros(len(labels)), labels))


def run_subject(graphs, subject: int, model_config: ModelConfig, config: TrainConfig, device,
                folds: int, split_seed: int) -> dict:
    """Cross-validate one subject. Every trial is tested exactly once."""
    labels = [int(graph.y) for graph in graphs]
    fold_results, confusion = [], np.zeros((2, 2), dtype=int)
    for number, (train_index, test_index) in enumerate(fold_indices(labels, folds, split_seed), start=1):
        result = run_fold([graphs[i] for i in train_index], [graphs[i] for i in test_index],
                          f"S{subject} fold {number}/{folds}", model_config, config, device)
        fold_results.append(dict(fold=number, **result))
        confusion += np.array(result["confusion_matrix"])
    # Scores of the subject are computed over all its trials, each predicted by
    # the fold that did not train on it.
    true = [0] * int(confusion[0].sum()) + [1] * int(confusion[1].sum())
    predicted = ([0] * int(confusion[0, 0]) + [1] * int(confusion[0, 1])
                 + [0] * int(confusion[1, 0]) + [1] * int(confusion[1, 1]))
    accuracies = [fold["accuracy"] for fold in fold_results]
    return dict(classification_metrics(true, predicted), fold_accuracy_min=float(min(accuracies)),
                fold_accuracy_max=float(max(accuracies)), folds=fold_results)


def run_within(data_dir, data_config: DataConfig = DataConfig(), model_config: ModelConfig | None = None,
               config: TrainConfig | None = None, subjects=None, folds: int = 10, split_seed: int = 42) -> dict:
    """Cross-validate every subject in `subjects` (default: all) separately."""
    config = config or within_config()
    config.validate()
    model_config = model_config or ModelConfig.for_data(data_config)
    device = select_device(config.device)
    subjects = tuple(subjects or data_config.subjects)
    if not set(subjects) <= set(data_config.subjects):
        raise ValueError("Subjects must be among the configured subjects")
    results = {}
    for subject in subjects:
        print(f"S{subject}: building graphs", flush=True)
        graphs = build_subject_graphs(data_dir, subject, data_config)
        results[str(subject)] = run_subject(graphs, subject, model_config, config, device, folds, split_seed)
        print(f"S{subject}: accuracy {results[str(subject)]['accuracy']:.4f} over {len(graphs)} trials", flush=True)
    summary = summarize(results)
    print(f"Mean accuracy over subjects: {summary['accuracy']:.4f}", flush=True)
    return dict(dataset=data_config.dataset, protocol="within_subject_cross_validation",
                evaluation="final epoch, once per fold; no selection on test trials",
                folds=folds, split_seed=split_seed, summary=summary, subjects=results,
                data_config=asdict(data_config), model_config=asdict(model_config), train_config=asdict(config),
                unlabeled_test_trial_use=dict(
                    alignment="Euclidean alignment uses all trials of the subject, including each test fold's",
                    center_logits=config.center_logits,
                    note="No labels of test trials are used."),
                device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                packages={name: version(name) for name in PACKAGES})
