"""Supervised learning with explicit separate or combined evaluation protocols."""
from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config_syn import SynTrainingConfig
from .engine import train_epoch, evaluate
from .metrics import compute_classification_metrics
from .setup import seed_fold, build_model, build_optimizer, build_criterion
from .splitting_syn import build_loaders, combine_evaluation_loaders, make_augmented_folds
from .experiment import run_fold
from .metrics import summarize_fold_results

if __package__ == 'train':
    from datacustom.dataset import build_subject_dataset, sha256
else:
    from ..datacustom.dataset import build_subject_dataset, sha256


def run_subject(partitions, device: torch.device, config: SynTrainingConfig) -> dict:
    config.validate()
    standard = config.standard()
    seed_fold(config.model_seed)
    training, validation, testing = build_loaders(partitions, config)
    selection_loader = (combine_evaluation_loaders(validation, testing, config)
                        if config.combine_valid_test else validation)
    model = build_model(partitions['training'][0].num_node_features, device, standard)
    optimizer = build_optimizer(model, standard)
    criterion = build_criterion(standard)
    best_accuracy, selected_epoch, best_state = -1.0, 0, None
    best_evaluation = None
    history = []
    subject = int(partitions['training'][0].subject_id)
    for epoch in tqdm(range(1, config.num_epochs + 1), desc=f'Training subject {subject}',
                      disable=not config.show_progress, leave=False):
        train_epoch(model, training, optimizer, criterion, device)
        train_accuracy = evaluate(model, training, device)
        if config.combine_valid_test:
            evaluation = evaluate(model, selection_loader, device, collect_predictions=True)
            selection_accuracy = evaluation.accuracy
            score_name = 'combined_test_accuracy'
        else:
            evaluation = None
            selection_accuracy = evaluate(model, selection_loader, device)
            score_name = 'validation_accuracy'
        history.append(dict(epoch=epoch, training_accuracy=train_accuracy, **{score_name: selection_accuracy}))
        if selection_accuracy > best_accuracy:
            selected_epoch, best_accuracy = epoch, selection_accuracy
            best_evaluation = evaluation
            # Keep a true copy, not references to tensors changed by later updates.
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    # Combined mode reports the selected epoch's collected predictions, just as
    # the original best-test trainer does; it is not an independent final test.
    test = best_evaluation if config.combine_valid_test else evaluate(model, testing, device, collect_predictions=True)
    assert test is not None
    counts = {name: {'total': len(graphs),
                     'original': sum(not bool(g.is_synthetic) for g in graphs),
                     'synthetic': sum(bool(g.is_synthetic) for g in graphs),
                     'class_counts': {str(label): sum(int(g.y.item()) == label for g in graphs) for label in (0, 1)}}
              for name, graphs in partitions.items()}
    score_key = 'best_combined_test_accuracy' if config.combine_valid_test else 'best_validation_accuracy'
    evaluation_graphs = (partitions['validation'] + partitions['testing']
                         if config.combine_valid_test else partitions['testing'])
    return dict(selected_epoch=selected_epoch, **{score_key: best_accuracy},
                test={'accuracy': test.accuracy, **compute_classification_metrics(test.labels, test.predictions)},
                evaluation_trial_count=len(evaluation_graphs),
                evaluation_sample_ids=[g.sample_id for g in evaluation_graphs],
                counts=counts, batches_per_epoch=len(training), optimizer_updates=len(training) * config.num_epochs,
                sample_ids={name: [g.sample_id for g in graphs] for name, graphs in partitions.items()},
                training_order=[g.sample_id for g in training.dataset], history=history)



def run_kfold_subject(partitions, device, config: SynTrainingConfig) -> dict:
    """Reuse the standard fold trainer, including its best-test selection rule."""
    folds = []
    subject = int(partitions['training'][0].subject_id)
    for number, (training, testing) in enumerate(make_augmented_folds(partitions, config), 1):
        result = dict(run_fold(training, testing, device, config.standard(), subject, number))
        result['counts'] = {
            name: {'total': len(graphs),
                   'original': sum(not bool(g.is_synthetic) for g in graphs),
                   'synthetic': sum(bool(g.is_synthetic) for g in graphs),
                   'class_counts': {str(c): sum(int(g.y.item()) == c for g in graphs) for c in (0, 1)}}
            for name, graphs in [('training', training), ('testing', testing)]}
        result['sample_ids'] = {name: [g.sample_id for g in graphs]
                                for name, graphs in [('training', training), ('testing', testing)]}
        result['batches_per_epoch'] = (len(training) + config.batch_size - 1) // config.batch_size
        result['optimizer_updates'] = result['batches_per_epoch'] * config.num_epochs
        folds.append(result)
    summary = summarize_fold_results(folds)
    result = dict(summary)
    result['test'] = {'accuracy': summary['mean'],
                      'balanced_accuracy': summary['balanced_accuracy']['mean'],
                      'macro_f1': summary['macro_f1']['mean'],
                      'confusion_matrix': summary['confusion_matrix']}
    return result

def run_experiment(original_dir, synthetic_dir, device, dataset_config, training_config: SynTrainingConfig) -> dict:
    training_config.validate()
    original = Path(original_dir).resolve()
    synthetic = Path(synthetic_dir).resolve() if synthetic_dir is not None else None
    sources = {'original_directory': str(original),
               'original': {name: sha256(original / name) for name in ('metadata.json', 'partitions.csv')}}
    if synthetic is not None:
        sources['synthetic_directory'] = str(synthetic)
        sources['synthetic'] = {name: sha256(synthetic / name) for name in ('metadata.json', 'manifest.csv')}
    subjects = {}
    for subject in training_config.subjects:
        partitions = build_subject_dataset(original, synthetic, subject, dataset_config)
        result = (run_kfold_subject(partitions, device, training_config) if training_config.protocol == "kfold"
                  else run_subject(partitions, device, training_config))
        result['source_sha256'] = {'original': sha256(original / f'subject_{subject:02d}.npz')}
        if synthetic is not None:
            result['source_sha256']['synthetic'] = sha256(synthetic / f'subject_{subject:02d}.npz')
        subjects[str(subject)] = result
        if training_config.protocol == 'kfold':
            print(f'S{subject}: Mean: {result["mean"]:.4f}, Max: {result["max"]:.4f}, Min: {result["min"]:.4f}', flush=True)
        elif training_config.combine_valid_test:
            print(f'S{subject}: selected epoch {result["selected_epoch"]}, '
                  f'best combined test accuracy {result["test"]["accuracy"]:.4f} '
                  f'({result["evaluation_trial_count"]} original trials)', flush=True)
        else:
            print(f'S{subject}: selected epoch {result["selected_epoch"]}, '
                  f'validation accuracy {result["best_validation_accuracy"]:.4f}, '
                  f'test accuracy {result["test"]["accuracy"]:.4f}', flush=True)
    summary = {key: float(np.mean([result['test'][key] for result in subjects.values()]))
               for key in ('accuracy', 'balanced_accuracy', 'macro_f1')}
    summary['confusion_matrix'] = np.sum([result['test']['confusion_matrix'] for result in subjects.values()], axis=0).tolist()
    summary['max_accuracy'] = max(result['test']['accuracy'] for result in subjects.values())
    summary['min_accuracy'] = min(result['test']['accuracy'] for result in subjects.values())
    print(f'Across subjects: Mean: {summary["accuracy"]:.4f}, '
          f'Max: {summary["max_accuracy"]:.4f}, Min: {summary["min_accuracy"]:.4f}')
    results = dict(protocol='fixed_within_subject', condition='original_plus_synthetic' if synthetic else 'original_only',
                selection_policy='best_combined_test_accuracy' if training_config.combine_valid_test else 'best_validation_accuracy',
                evaluation_partitions=['validation', 'testing'] if training_config.combine_valid_test else ['testing'],
                independent_final_test=not training_config.combine_valid_test,
                test_evaluations_per_subject=training_config.num_epochs if training_config.combine_valid_test else 1,
                class_labels=[0, 1], summary=summary, subjects=subjects,
                training_config=asdict(training_config), dataset_config=json.loads(json.dumps(asdict(dataset_config))),
                optimizer=training_config.standard().optimizer, loss=training_config.standard().loss,
                partition_policy='reuse saved assignments; no KFold or legacy interleaving',
                training_mix='one-time SHA-256 order keyed by mix_seed, subject_id, sample_id',
                checkpoint_storage='selected model state held in memory; graphs are never saved',
                device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                packages={name: version(name) for name in ('torch', 'torch-geometric', 'numpy', 'scipy', 'scikit-learn')},
                sources=sources)

    if training_config.protocol == 'kfold':
        results.update(
            protocol='within_subject_original_kfold_with_synthetic_training',
            selection_policy='best_test_accuracy', independent_final_test=False,
            evaluation_partitions=['original_training', 'original_validation', 'original_testing'],
            test_evaluations_per_subject=training_config.n_folds * training_config.num_epochs,
            partition_policy='KFold on all originals only; all synthetic trials appended to every training fold',
            original_order='session_id then original trial_id; independent of saved partition membership',
            training_mix='original fold order followed by synthetic manifest order; standard loader shuffle setting',
            checkpoint_storage='selected epoch metrics retained by standard trainer; no weights or graphs saved',
            synthetic_reused_across_folds=synthetic is not None,
            known_limitations=(['Existing diffusion model used original trials that can enter test folds; graph conversion does not remove this leakage.'] if synthetic else []) +
                ['Test-fold accuracy selects the epoch; results are not independent final test estimates.'])
    return results
