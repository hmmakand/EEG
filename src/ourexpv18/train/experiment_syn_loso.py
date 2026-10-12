"""Train saved LOSO folds; evaluate held-out subjects once at the final epoch."""
from dataclasses import asdict
from importlib.metadata import version
import json
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .config_syn import SynTrainingConfig
from .engine import train_epoch, evaluate
from .metrics import compute_classification_metrics, summarize_fold_results
from .reporting import print_subject_result
from .setup import seed_fold, build_model, build_optimizer, build_criterion, sample_dimensions

from ..datacustom.dataset import FoldGraphs, counts, sha256
from ..datacustom.loso import SavedLOSO

def run_fold(fold: FoldGraphs, device, config: SynTrainingConfig) -> dict[str, Any]:
    config.validate()
    learning = config.standard()
    seed_fold(config.model_seed)
    training, testing = fold.loaders(config.batch_size, config.train_shuffle)
    model = build_model(fold.training[0].num_node_features, device, learning,
                        **sample_dimensions(fold.training[0]))
    optimizer = build_optimizer(model, learning)
    criterion = build_criterion(learning)
    for epoch in tqdm(range(1, config.num_epochs + 1),
                      desc=f'Training LOSO held-out S{fold.subject}',
                      leave=False, disable=not config.show_progress):
        train_epoch(model, training, optimizer, criterion, device)
    test = evaluate(model, testing, device, collect_predictions=True)
    metrics = compute_classification_metrics(test.labels, test.predictions)
    return dict(fold=fold.fold, selected_epoch=config.num_epochs, accuracy=test.accuracy,
                **metrics,
                counts={name: counts(graphs) for name, graphs in [('training', fold.training), ('testing', fold.testing)]},
                sample_ids={name: [g.sample_id for g in graphs] for name, graphs in [('training', fold.training), ('testing', fold.testing)]},
                batches_per_epoch=len(training), optimizer_updates=len(training) * config.num_epochs,
                source_sha256=fold.source_sha256)


def run_experiment(bundle, device, dataset_config, training_config: SynTrainingConfig) -> dict:
    training_config.validate()
    data = SavedLOSO(bundle, dataset_config)
    subjects = training_config.subjects if training_config.subjects is not None else data.subjects
    if not set(subjects).issubset(data.subjects):
        raise ValueError('Requested subjects are not in the saved LOSO protocol')
    results = {}
    for subject in subjects:
        folds = []
        for number in data.fold_numbers(subject):
            print(f'S{subject} fold {number}: building graphs from saved fold', flush=True)
            fold = data.load_fold(subject, number)
            result = run_fold(fold, device, training_config)
            print(f'S{subject} fold {number}: selected epoch {result["selected_epoch"]}, '
                  f'test accuracy {result["accuracy"]:.4f}', flush=True)
            folds.append(result)
            del fold
        results[str(subject)] = summarize_fold_results(folds)
        print_subject_result(subject, results[str(subject)])
    summary = dict(accuracy=float(np.mean([r['mean'] for r in results.values()])),
                   balanced_accuracy=float(np.mean([r['balanced_accuracy']['mean'] for r in results.values()])),
                   macro_f1=float(np.mean([r['macro_f1']['mean'] for r in results.values()])),
                   confusion_matrix=np.sum([r['confusion_matrix'] for r in results.values()], axis=0).tolist())
    print(f'Overall mean subject accuracy: {summary["accuracy"]:.4f}', flush=True)
    return dict(protocol='saved_leave_one_subject_out', condition='original_plus_training_subject_synthetic',
                selection_policy='final_epoch', independent_final_test=True, class_labels=[0, 1],
                summary=summary, subjects=results,
                training_config=asdict(training_config), dataset_config=json.loads(json.dumps(asdict(dataset_config))),
                executed_subjects=list(subjects), saved_fold_counts={str(s): len(data.fold_numbers(s)) for s in subjects},
                saved_split_config=data.metadata['config'],
                optimizer=training_config.standard().optimizer, loss=training_config.standard().loss,
                partition_policy='load finalized manifest in saved row order; no runtime splitting or mixing',
                generator_test_isolation='held-out subject originals and generators excluded from classifier training',
                known_limitations=['Single model seed; no separate classifier validation set.'],
                checkpoint_storage='final epoch metrics only; no graphs or model weights saved',
                sources={'bundle': str(data.root), 'manifest': str(data.manifest_path.relative_to(data.root)),
                         'manifest_sha256': data.manifest_sha256, 'metadata_sha256': sha256(data.root / 'metadata.json'),
                         'jobs_sha256': sha256(data.root / 'jobs.json')},
                device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                packages={name: version(name) for name in ('torch', 'torch-geometric', 'numpy', 'scipy', 'scikit-learn')})
