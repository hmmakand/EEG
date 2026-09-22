"""Read only original training signal rows and persist the internal 90/10 split."""
import csv
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config
from .runtime import sha256

IDENTITY = ('source_file', 'subject_id', 'session_id', 'original_trial_index')


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError('Cannot save an empty manifest')
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def training_rows(original: Path, subject: int) -> list[dict[str, str]]:
    rows = [r for r in read_rows(original / 'partitions.csv')
            if int(r['subject_id']) == subject and r['partition'] == 'training']
    rows.sort(key=lambda r: int(r['partition_index']))
    if not rows or len({tuple(r[k] for k in IDENTITY) for r in rows}) != len(rows):
        raise ValueError(f'Missing or duplicate original training identities for subject {subject}')
    if len({int(r['array_index']) for r in rows}) != len(rows):
        raise ValueError('Original training array indices must be unique')
    return rows


def original_fingerprints(original: Path, config: Config) -> dict[str, str]:
    """Hash source artifacts without loading validation/test signal arrays."""
    metadata = json.loads((original / 'metadata.json').read_text())
    files = ['partitions.csv', *[f'subject_{s:02d}.npz' for s in config.subjects]]
    hashes = {'metadata.json': sha256(original / 'metadata.json')}
    for name in files:
        hashes[name] = sha256(original / name)
        if hashes[name] != metadata['artifact_sha256'][name]:
            raise ValueError(f'Original artifact does not match its manifest: {name}')
    return hashes


def prepare_internal_partitions(original: Path, run: Path, config: Config) -> None:
    path = run / 'internal_partitions.csv'
    if path.exists():
        validate_internal_partitions(original, path, config)
        return
    rows = []
    for subject in config.subjects:
        source = training_rows(original, subject)
        parts: dict[str, list[int]] = {'training': [], 'validation': []}
        for label in range(config.num_classes):
            indices = [i for i, r in enumerate(source) if int(r['class_label']) == label]
            if len(indices) <= config.validation_per_class:
                raise ValueError(f'Subject {subject}, class {label}: too few training trials')
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.split_seed, subject, 0, label])))
            rng.shuffle(indices)
            parts['validation'].extend(indices[:config.validation_per_class])
            parts['training'].extend(indices[config.validation_per_class:])
        for i, (subset, indices) in enumerate(parts.items()):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.split_seed, subject, 1, i])))
            rng.shuffle(indices)
            for position, index in enumerate(indices):
                rows.append({**source[index], 'diffusion_subset': subset,
                             'diffusion_index': str(position), 'internal_split_seed': str(config.split_seed)})
    write_rows(path, rows)
    validate_internal_partitions(original, path, config)


def validate_internal_partitions(original: Path, path: Path, config: Config) -> None:
    saved = read_rows(path)
    if {int(r['subject_id']) for r in saved} != set(config.subjects):
        raise ValueError('Internal manifest subjects differ from configuration')
    for subject in config.subjects:
        source = {tuple(r[k] for k in IDENTITY): r for r in training_rows(original, subject)}
        rows = [r for r in saved if int(r['subject_id']) == subject]
        keys = [tuple(r[k] for k in IDENTITY) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(source):
            raise ValueError('Internal subsets must partition original training trials exactly once')
        for row, key in zip(rows, keys):
            if any(row[k] != value for k, value in source[key].items()):
                raise ValueError('Internal manifest changed original trial information')
            if row['internal_split_seed'] != str(config.split_seed) or row['diffusion_subset'] not in ('training', 'validation'):
                raise ValueError('Invalid internal split seed or subset')
        for subset in ('training', 'validation'):
            selected = [r for r in rows if r['diffusion_subset'] == subset]
            if sorted(int(r['diffusion_index']) for r in selected) != list(range(len(selected))):
                raise ValueError('Invalid internal partition order')
            for label in range(config.num_classes):
                count = sum(int(r['class_label']) == label for r in selected)
                if (subset == 'validation' and count != config.validation_per_class) or count == 0:
                    raise ValueError('Internal partition class counts are invalid')


def selected_signals(path: Path, indices: list[int]) -> np.ndarray:
    """Read selected C-order NPY rows from the NPZ stream, not the full signal array.

    ZIP may decompress skipped bytes internally; no held-out trial is materialized
    as an EEG array or supplied to a model.
    """
    with zipfile.ZipFile(path) as archive, archive.open('signals.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported NPY version: {version}')
        if fortran or dtype.hasobject or len(shape) != 3:
            raise ValueError('Expected a numeric C-order [trial, sample, channel] array')
        if not indices or min(indices) < 0 or max(indices) >= shape[0] or len(set(indices)) != len(indices):
            raise ValueError('Invalid or duplicate signal row indices')
        start = stream.tell()
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        result = np.empty((len(indices), *shape[1:]), dtype=dtype)
        for position, index in sorted(enumerate(indices), key=lambda item: item[1]):
            stream.seek(start + index * row_bytes)
            raw = stream.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError('Truncated EEG signal array')
            result[position] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
    return result


def subject_loaders(original: Path, run: Path, subject: int, config: Config):
    rows = [r for r in read_rows(run / 'internal_partitions.csv') if int(r['subject_id']) == subject]
    signals = selected_signals(original / f'subject_{subject:02d}.npz', [int(r['array_index']) for r in rows])
    if signals.shape[1:] != (config.time_points, config.in_channels) or not np.isfinite(signals).all():
        raise ValueError('Signal shape or values are invalid')
    # Division before float32 conversion mirrors the reference collate pipeline.
    x = torch.from_numpy(np.ascontiguousarray((signals / config.signal_scale).transpose(0, 2, 1), dtype=np.float32))
    y = torch.tensor([int(r['class_label']) for r in rows], dtype=torch.long)
    loaders = []
    for subset in ('training', 'validation'):
        indices = sorted((i for i, r in enumerate(rows) if r['diffusion_subset'] == subset),
                         key=lambda i: int(rows[i]['diffusion_index']))
        loaders.append(DataLoader(TensorDataset(x[indices], y[indices]), batch_size=config.batch_size,
                                  shuffle=subset == 'training', num_workers=0))
    return loaders[0], loaders[1]
