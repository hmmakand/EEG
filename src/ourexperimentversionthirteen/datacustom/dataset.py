"""Adapt saved original/synthetic EEG to the existing graph pipeline; never resplit."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from torch_geometric.data import Data

if __package__ == 'datacustom':
    from dataset import DatasetConfig, default_dataset_config, build_trial_graph
    from dataset.trials import Trial
    from dataset.validation import validate_config
else:
    from ..dataset import DatasetConfig, default_dataset_config, build_trial_graph
    from ..dataset.trials import Trial
    from ..dataset.validation import validate_config

PARTITIONS = ('training', 'validation', 'testing')
ORIGINAL = Path(__file__).resolve().parents[1] / 'datasyn' / 'original'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path, subject: int) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return [row for row in csv.DictReader(stream) if int(row['subject_id']) == subject]


def verify_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'Saved artifact has changed: {path}')
    return actual


def validate_arrays(signals, labels, config: DatasetConfig) -> None:
    shape = (int(config.sample_rate * config.trials.duration_seconds), len(config.trials.channel_indices))
    if signals.ndim != 3 or signals.shape[1:] != shape or len(labels) != len(signals) or labels.ndim != 1:
        raise ValueError(f'Expected signals [trials, {shape[0]}, {shape[1]}] and one label per trial')
    if not np.isfinite(signals).all() or not np.issubdtype(labels.dtype, np.integer) or not set(labels.tolist()) <= {0, 1}:
        raise ValueError('Signals must be finite and labels must be binary integers')


def validate_manifest(rows, labels, *, synthetic: bool) -> None:
    if sorted(int(row['array_index']) for row in rows) != list(range(len(labels))):
        raise ValueError('Manifest must cover every saved trial exactly once')
    for row in rows:
        if int(row['class_label']) != int(labels[int(row['array_index'])]):
            raise ValueError('Manifest label disagrees with saved array')
        partition = row['intended_partition'] if synthetic else row['partition']
        if partition not in PARTITIONS or (synthetic and partition != 'training'):
            raise ValueError('Synthetic trials are allowed only in training')
    identity_keys = ('synthetic_trial_id',) if synthetic else ('source_file', 'session_id', 'original_trial_index')
    if len({tuple(row[k] for k in identity_keys) for row in rows}) != len(rows):
        raise ValueError('Duplicate trial identities in manifest')


def build_subject_dataset(original_dir: str | Path = ORIGINAL, synthetic_dir: str | Path | None = None,
                          subject_number: int = 1, dataset_config: DatasetConfig | None = None) -> dict[str, list[Data]]:
    """Return training/validation/testing PyG lists; synthetic data is optional."""
    config = dataset_config or default_dataset_config()
    validate_config(config)
    original = Path(original_dir)
    metadata = json.loads((original / 'metadata.json').read_text())
    extraction = metadata['config']
    if extraction['sample_rate'] != config.sample_rate:
        raise ValueError('Graph sampling rate must match original extraction')
    for name in ('channel_indices', 'session_indices', 'label_map', 'duration_seconds', 'event_index_offset'):
        expected = json.loads(json.dumps(getattr(config.trials, name)))
        if extraction['trials'][name] != expected:
            raise ValueError(f'Graph configuration differs from saved extraction: {name}')
    filename = f'subject_{subject_number:02d}.npz'
    hashes = {'metadata.json': sha256(original / 'metadata.json')}
    for name in ('partitions.csv', filename):
        hashes[name] = verify_hash(original / name, metadata['artifact_sha256'][name])
    rows = read_rows(original / 'partitions.csv', subject_number)
    with np.load(original / filename, allow_pickle=False) as data:
        signals, labels = data['signals'], data['labels']
        validate_arrays(signals, labels, config)
        validate_manifest(rows, labels, synthetic=False)
        for row in rows:
            index = int(row['array_index'])
            if (int(data['subject_ids'][index]) != subject_number
                or int(data['session_ids'][index]) != int(row['session_id'])
                or int(data['original_trial_indices'][index]) != int(row['original_trial_index'])):
                raise ValueError('Original array identity disagrees with manifest')
    partitions: dict[str, list[Data]] = {name: [] for name in PARTITIONS}
    for name in PARTITIONS:
        selected = sorted((r for r in rows if r['partition'] == name), key=lambda r: int(r['partition_index']))
        if not selected or [int(r['partition_index']) for r in selected] != list(range(len(selected))):
            raise ValueError(f'Missing or invalid saved partition order: {name}')
        if {int(r['class_label']) for r in selected} != {0, 1}:
            raise ValueError(f'Both classes are required in {name}')
        for row in selected:
            index = int(row['array_index'])
            trial = Trial(signals[index], int(labels[index]), subject_number,
                          int(row['session_id']), int(row['original_trial_index']))
            graph = build_trial_graph(trial, config)
            graph.is_synthetic = False
            graph.sample_id = f"original:{row['source_file']}:{row['session_id']}:{row['original_trial_index']}"
            graph.partition = name
            graph.source_run = ''
            partitions[name].append(graph)
    if synthetic_dir is not None:
        synthetic = Path(synthetic_dir)
        synth_metadata = json.loads((synthetic / 'metadata.json').read_text())
        if synth_metadata.get('status') != 'complete':
            raise ValueError('Select a completed synthetic run explicitly')
        for name, digest in hashes.items():
            if synth_metadata['original_sha256'].get(name) != digest:
                raise ValueError(f'Synthetic run was generated from different original data: {name}')
        if synth_metadata.get('original_partition_used') != 'training' or synth_metadata.get('original_validation_and_testing_used') is not False:
            raise ValueError('Synthetic run must use only original training trials')
        generation = synth_metadata['generation'][str(subject_number)]
        verify_hash(synthetic / filename, generation['sha256'])
        verify_hash(synthetic / 'manifest.csv', synth_metadata['manifest_sha256'])
        synth_rows = sorted(read_rows(synthetic / 'manifest.csv', subject_number), key=lambda r: int(r['array_index']))
        with np.load(synthetic / filename, allow_pickle=False) as data:
            signals, labels = data['signals'], data['labels']
            validate_arrays(signals, labels, config)
            validate_manifest(synth_rows, labels, synthetic=True)
            if len(labels) != generation['trial_count'] or list(signals.shape) != generation['shape']:
                raise ValueError('Synthetic array size disagrees with generation metadata')
            for row in synth_rows:
                index = int(row['array_index'])
                if (int(data['subject_ids'][index]) != subject_number or str(data['synthetic_trial_ids'][index]) != row['synthetic_trial_id']
                    or row['output_file'] != filename or row['source'] != 'synthetic'
                    or row['checkpoint_sha256'] != generation['checkpoint_sha256']):
                    raise ValueError('Synthetic identity or provenance disagrees with manifest')
                trial = Trial(signals[index], int(labels[index]), subject_number, -1, index)
                graph = build_trial_graph(trial, config)
                graph.is_synthetic = True
                graph.sample_id = f"synthetic:{synthetic.name}:{row['synthetic_trial_id']}"
                graph.partition = 'training'
                graph.source_run = synthetic.name
                partitions['training'].append(graph)
    return partitions
