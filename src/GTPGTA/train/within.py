"""Nested within-subject folds with training-only alignment and fresh final refits."""
from dataclasses import replace
from pathlib import Path
import warnings

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold

from ..datascripts import DataConfig, read_samples, build_graph, liu2024_config
from ..datascripts.signal import alignment_matrix
from ..model import ModelConfig
from ..model.config import PATHS
from .config import TrainConfig
from .loso import run_cost, run_fold, run_information, select_device
from .metrics import classification_metrics, format_metrics, print_summary, summarize
from .validation import select_epochs

WITHIN_DEFAULTS = dict(epochs=100, center_logits=False, target_calibration=False)


def within_config(**overrides):
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


def nested_indices(trial_ids, labels, folds=10, split_seed=42):
    if folds < 3:
        raise ValueError("Validation requires at least three folds")
    outer = fold_indices(trial_ids, labels, folds, split_seed)
    for k, (non_test, testing) in enumerate(outer):
        validation = outer[(k + 1) % folds][1]
        training = np.setdiff1d(non_test, validation)
        yield training, validation, testing, non_test


def partition_graphs(samples, fitting_indices, chosen_indices, config):
    """Fit alignment on training samples only; apply the frozen transform to each partition."""
    matrix = alignment_matrix([samples[i] for i in fitting_indices], config)
    return {int(i): build_graph(replace(samples[i], signal=samples[i].signal @ matrix), config)
            for i in chosen_indices}


def run_subject(samples, subject, data_config, model_config, config, device, folds, split_seed, models_dir=None):
    labels, trials = [s.label for s in samples], [s.trial_id for s in samples]
    fold_results, first_graphs = [], None
    splits = list(nested_indices(trials, labels, folds, split_seed))
    for k, (train_index, val_index, test_index, non_test) in enumerate(splits):
        selected = partition_graphs(samples, train_index, non_test, data_config)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        selection = select_epochs([selected[int(i)] for i in train_index],
                                  {subject: [selected[int(i)] for i in val_index]},
                                  model_config, config, device, f"S{subject} fold {k + 1} validation")
        selection_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        selection['split'] = dict(training_index=train_index.tolist(), validation_index=val_index.tolist(),
                                  test_index=test_index.tolist(), validation_fold=(k + 1) % folds + 1,
                                  test_fold=k + 1, split_seed=split_seed)
        selection['trial_assignments'] = {
            name: sorted({int(trials[i]) for i in indices})
            for name, indices in [('training', train_index), ('validation', val_index), ('test', test_index)]}
        final_graphs = partition_graphs(samples, non_test, range(len(samples)), data_config)
        if first_graphs is None:
            first_graphs = list(final_graphs.values())
        result = run_fold([final_graphs[int(i)] for i in non_test], [final_graphs[int(i)] for i in test_index],
                          f"S{subject} fold {k + 1}/{folds}", data_config, model_config,
                          replace(config, epochs=selection['selected_epoch']), device,
                          models_dir / f"S{subject}_fold{k + 1}" if models_dir else None, selection)
        if selection_peak is not None:
            result['peak_gpu_memory_bytes'] = max(result['peak_gpu_memory_bytes'], selection_peak)
        result['refit_index'] = non_test.tolist()
        fold_results.append(dict(fold=k + 1, test_index=test_index.tolist(), **result))
    pooled = lambda key: [value for fold in fold_results for value in fold[key]]
    features = {name: np.concatenate([fold['embeddings'][name] for fold in fold_results])
                for name in fold_results[0]['embeddings']}
    for fold in fold_results:
        del fold['embeddings']
    accuracies = [fold['accuracy'] for fold in fold_results]
    return (dict(classification_metrics(pooled('labels'), pooled('predictions'), pooled('scores')),
                 labels=pooled('labels'), predictions=pooled('predictions'), scores=pooled('scores'),
                 fold_accuracy_min=float(min(accuracies)), fold_accuracy_max=float(max(accuracies)),
                 folds=fold_results, embeddings=features, samples=len(samples)), first_graphs)


def run_within(data_dir, data_config=None, config=None, subjects=None, folds=10, split_seed=42,
               models_dir=None, paths=PATHS):
    data_config = data_config or liu2024_config()
    config = config or within_config()
    config.validate()
    if folds < 3:
        raise ValueError('Within-subject validation requires at least three folds')
    if config.target_calibration or config.center_logits:
        raise ValueError('Within-subject evaluation uses neither target calibration nor logit centering')
    caveat = None
    if data_config.dataset == 'jia2019':
        caveat = ('Jia2019 within-subject results are confounded: each class is stored in a separate recording. '
                  'Validation does not remove this recording/class confound; not a valid MI comparison.')
        warnings.warn(caveat, UserWarning, stacklevel=2)
    model_config = ModelConfig.for_data(data_config, paths=tuple(paths))
    device = select_device(config.device)
    subjects = tuple(subjects or data_config.subjects)
    if not set(subjects) <= set(data_config.subjects):
        raise ValueError('Subjects must be among the configured subjects')
    results, first_graphs = {}, None
    for subject in subjects:
        samples = read_samples(data_dir, subject, data_config)
        result, graphs = run_subject(samples, subject, data_config, model_config, config, device,
                                     folds, split_seed, models_dir)
        first_graphs = first_graphs or graphs
        results[str(subject)] = result
        print(f"S{subject}: {format_metrics(result)}", flush=True)
    summary = summarize(results)
    print_summary(f"Mean over {len(results)} subjects", summary)
    features = {subject: result.pop('embeddings') for subject, result in results.items()}
    return dict(protocol='within_subject_cross_validation',
                evaluation='Validation-only early stopping; fresh refit on all non-test folds for the selected '
                           'epoch count; evaluate each outer test fold once.',
                validity=dict(valid_for_mi_comparison=caveat is None, caveat=caveat),
                folds=folds, split_seed=split_seed, summary=summary, subjects=results,
                unlabeled_test_sample_use=dict(alignment='Fitted on training folds only, then frozen; refitted '
                    'on all non-test folds for final training.', target_calibration=False, center_logits=False),
                cost=run_cost([fold for result in results.values() for fold in result['folds']]),
                embeddings=features, **run_information(data_config, model_config, config, device, first_graphs))
