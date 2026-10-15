"""Within-subject cross-validation: train and test on samples of the same subject.

Every subject is evaluated alone. Its trials are split into folds; for each
fold the ensemble is trained on the other folds and the fold is scored once,
after the final epoch. Nothing is selected on test samples: the number of
epochs is fixed in advance (see WITHIN_DEFAULTS).

Differences from the leave-one-subject-out run:
  - No target calibration and no logit centring: the model was trained on the
    subject it is tested on, so there is no new subject to adapt to.
  - Per-subject alignment is computed from all samples of the subject,
    including the test fold's, without their labels.

Folds are made of trials. For liu2024 the imagery and the rest sample of a
trial are therefore always in the same fold. For bci2a, which has one sample
per trial, the folds are stratified by class.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..datascripts import DataConfig, build_subject_graphs, liu2024_config
from ..model import ModelConfig
from ..model.config import PATHS
from .config import TrainConfig
from .loso import run_cost, run_fold, run_information, select_device
from .metrics import classification_metrics, format_metrics, print_summary, summarize

# Training settings that differ from the leave-one-subject-out defaults. A fold
# has few training samples (about 130 for bci2a, 72 for liu2024), so more epochs
# are needed; 60 was chosen in version 23 on validation splits taken from
# training folds of bci2a, and was not re-tuned for liu2024.
WITHIN_DEFAULTS = dict(epochs=60, center_logits=False, target_calibration=False)


def within_config(**overrides) -> TrainConfig:
    return replace(TrainConfig(), **{**WITHIN_DEFAULTS, **overrides})


def fold_indices(trial_ids, labels, folds: int, split_seed: int):
    """Shuffled folds of trials: [(training sample indices, testing sample indices), ...].

    All samples of a trial go to the same fold. Trials with a single sample
    are stratified by its label.
    """
    if folds < 2:
        raise ValueError("At least two folds are required")
    trial_ids, labels = np.asarray(trial_ids), np.asarray(labels)
    trials = np.unique(trial_ids)
    members = [np.flatnonzero(trial_ids == trial) for trial in trials]
    strata = [int(labels[rows[0]]) if len(rows) == 1 else 0 for rows in members]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    return [(np.sort(np.concatenate([members[i] for i in train])), np.sort(np.concatenate([members[i] for i in test])))
            for train, test in splitter.split(np.zeros(len(trials)), strata)]


def run_subject(graphs, subject: int, data_config: DataConfig, model_config: ModelConfig, config: TrainConfig, device,
                folds: int, split_seed: int, models_dir: Path | None = None) -> dict:
    """Cross-validate one subject. Every sample is tested exactly once."""
    labels = [int(graph.y) for graph in graphs]
    fold_results = []
    splits = fold_indices([graph.trial_id for graph in graphs], labels, folds, split_seed)
    for number, (train_index, test_index) in enumerate(splits, start=1):
        result = run_fold([graphs[i] for i in train_index], [graphs[i] for i in test_index],
                          f"S{subject} fold {number}/{folds}", data_config, model_config, config, device,
                          models_dir / f"S{subject}_fold{number}" if models_dir else None)
        fold_results.append(dict(fold=number, test_index=test_index.tolist(), **result))
    # Scores of the subject are computed over all its samples, each predicted by
    # the fold that did not train on it; the highest-epoch score takes the best
    # epoch of every fold.
    pooled = lambda key: [value for fold in fold_results for value in fold[key]]
    accuracies = [fold["accuracy"] for fold in fold_results]
    # Embeddings of every sample, from the fold that tested it, in the order of the pooled labels.
    features = {name: np.concatenate([fold["embeddings"][name] for fold in fold_results])
                for name in fold_results[0]["embeddings"]}
    for fold in fold_results:
        del fold["embeddings"]
    return dict(classification_metrics(pooled("labels"), pooled("predictions"), pooled("scores")),
                fold_accuracy_min=float(min(accuracies)), fold_accuracy_max=float(max(accuracies)),
                highest_epoch=classification_metrics(pooled("labels"), pooled("highest_epoch_predictions"),
                                                     pooled("highest_epoch_scores")),
                folds=fold_results, embeddings=features, samples=len(graphs))


def run_within(data_dir, data_config: DataConfig | None = None, config: TrainConfig | None = None, subjects=None,
               folds: int = 10, split_seed: int = 42, models_dir: Path | None = None, paths=PATHS) -> dict:
    """Cross-validate every subject in `subjects` (default: all) separately. `paths`: as in run_loso."""
    data_config = data_config or liu2024_config()
    config = config or within_config()
    config.validate()
    if config.target_calibration:
        raise ValueError("Target calibration is defined for leave one subject out only")
    model_config = ModelConfig.for_data(data_config, paths=tuple(paths))
    device = select_device(config.device)
    subjects = tuple(subjects or data_config.subjects)
    if not set(subjects) <= set(data_config.subjects):
        raise ValueError("Subjects must be among the configured subjects")
    results, first_graphs = {}, None
    for subject in subjects:
        print(f"S{subject}: building graphs", flush=True)
        graphs = build_subject_graphs(data_dir, subject, data_config)
        first_graphs = first_graphs or graphs
        results[str(subject)] = run_subject(graphs, subject, data_config, model_config, config, device, folds,
                                            split_seed, models_dir)
        result = results[str(subject)]
        print(f"S{subject} ({len(graphs)} samples): {format_metrics(result)} "
              f"(highest test epoch per fold: acc {result['highest_epoch']['accuracy']:.4f})", flush=True)
    summary = summarize(results)
    summary["highest_epoch"] = summarize({k: v["highest_epoch"] for k, v in results.items()})
    print_summary(f"Mean over {len(results)} subjects", summary)
    features = {subject: result.pop("embeddings") for subject, result in results.items()}
    return dict(protocol="within_subject_cross_validation",
                evaluation="The result is the score after the final epoch of every fold. highest_epoch takes the "
                           "epoch with the highest test accuracy in every fold, chosen with the test labels; it is "
                           "biased upward and recorded only for comparison.",
                folds=folds, split_seed=split_seed, summary=summary, subjects=results,
                unlabeled_test_sample_use=dict(
                    alignment="Euclidean alignment uses all samples of the subject, including each test fold's",
                    note="No labels of test samples are used."),
                cost=run_cost([fold for result in results.values() for fold in result["folds"]]),
                embeddings=features, **run_information(data_config, model_config, config, device, first_graphs))
