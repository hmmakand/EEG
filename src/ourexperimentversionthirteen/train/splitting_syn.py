"""Fixed-partition loaders and original-only folds with synthetic augmentation."""
import hashlib

import torch
from torch_geometric.loader import DataLoader

from .config_syn import SynTrainingConfig


def build_loaders(partitions, config: SynTrainingConfig):
    validate_partitions(partitions)
    config.validate()
    return _fixed_loaders(partitions, config)


def validate_partitions(partitions):
    if set(partitions) != {'training', 'validation', 'testing'}:
        raise ValueError('Expected training, validation, and testing graph lists')
    seen = set()
    subjects = set()
    for name, graphs in partitions.items():
        if not graphs:
            raise ValueError(f'Empty partition: {name}')
        for graph in graphs:
            if graph.partition != name:
                raise ValueError('Graph partition does not match its assigned list')
            if graph.sample_id in seen:
                raise ValueError('Trial identity appears more than once across partitions')
            if bool(graph.is_synthetic) and name != 'training':
                raise ValueError('Synthetic graphs cannot enter validation or testing')
            seen.add(graph.sample_id)
            subjects.add(int(graph.subject_id))
    if len(subjects) != 1:
        raise ValueError('Loaders must contain exactly one subject')


def _fixed_loaders(partitions, config):
    # Mix once without changing list membership or the global model RNG.
    # Per-ID ordering preserves the real trials' relative order in both conditions.
    training = sorted(partitions['training'], key=lambda g: hashlib.sha256(
        f'{config.mix_seed}:{g.subject_id}:{g.sample_id}'.encode()).digest())
    generator = torch.Generator().manual_seed(config.mix_seed)
    return (DataLoader(training, batch_size=config.batch_size, shuffle=config.train_shuffle, generator=generator),
            DataLoader(partitions['validation'], batch_size=config.batch_size, shuffle=False),
            DataLoader(partitions['testing'], batch_size=config.batch_size, shuffle=False))


def combine_evaluation_loaders(validation, testing, config: SynTrainingConfig):
    """Combine already-validated held-out lists without changing saved assignments."""
    return DataLoader(list(validation.dataset) + list(testing.dataset),
                      batch_size=config.batch_size, shuffle=False)


def make_augmented_folds(partitions, config: SynTrainingConfig):
    """Split originals only; append the same synthetic graphs to every train fold."""
    from .splitting import make_fold_indices, select_fold_data

    config.validate()
    validate_partitions(partitions)
    graphs = [g for name in ('training', 'validation', 'testing') for g in partitions[name]]
    # Canonical identity order is independent of the old saved partition order.
    original = sorted((g for g in graphs if not bool(g.is_synthetic)),
                      key=lambda g: (int(g.session_id), int(g.trial_id)))
    synthetic = [g for g in graphs if bool(g.is_synthetic)]
    for training_indices, testing_indices in make_fold_indices(original, config.standard()):
        training, testing = select_fold_data(original, training_indices, testing_indices)
        yield training + synthetic, testing
