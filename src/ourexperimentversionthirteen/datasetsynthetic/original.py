"""Save unfiltered original EEG segments and reusable 70/15/15 partitions."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, TypedDict

import numpy as np

if __package__ in (None, '', 'datasetsynthetic'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset.data_io import load_mat_file, read_sessions, resolve_data_dir, subject_file_path
    from dataset.trials import extract_trials
    from datasetsynthetic.config import DEFAULT_OUTPUT, PARTITIONS, SplitConfig
else:
    from ..dataset.data_io import load_mat_file, read_sessions, resolve_data_dir, subject_file_path
    from ..dataset.trials import extract_trials
    from .config import DEFAULT_OUTPUT, PARTITIONS, SplitConfig


FIELDS = ('source_file', 'subject_id', 'session_id', 'original_trial_index',
          'original_event_position', 'start_sample', 'original_class_label',
          'class_label', 'array_index', 'partition', 'partition_index', 'split_seed')


class SplitMetadata(TypedDict):
    format_version: int
    config: dict[str, Any]
    source_sha256: dict[str, str]
    source_directory: str
    numpy_version: str
    signal_axes: list[str]
    processing: str
    rounding: str
    randomization: str
    index_convention: str
    counts: dict[str, dict[str, dict[str, int]]]
    exclusions: list[dict[str, Any]]
    artifact_sha256: dict[str, str]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def partition_counts(count, percentages=(70, 15, 15)):
    numerators = np.asarray(percentages, dtype=np.int64) * count
    counts = numerators // 100
    order = sorted(range(3), key=lambda i: (-(numerators[i] % 100), i))
    for i in order[:count - int(counts.sum())]:
        counts[i] += 1
    if np.any(counts == 0):
        raise ValueError(f'{count} trials in a class cannot populate all partitions with these ratios')
    return counts.tolist()


def split_indices(labels, subject_id, config):
    """Independent PCG64 streams derived from one seed, subject, and class."""
    labels = np.asarray(labels)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Each subject must contain both left and right classes')
    parts = {name: [] for name in PARTITIONS}
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        counts = partition_counts(len(indices), config.percentages)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.seed, subject_id, 0, label])))
        rng.shuffle(indices)
        offset = 0
        for name, count in zip(PARTITIONS, counts):
            parts[name].extend(indices[offset:offset + count].tolist())
            offset += count
    for i, name in enumerate(PARTITIONS):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([config.seed, subject_id, 1, i])))
        rng.shuffle(parts[name])
    return parts


def extract_original(path, subject, config):
    sessions = list(read_sessions(load_mat_file(path)))
    trials = extract_trials(sessions, subject, config.sample_rate, config.trials)
    # Canonical original order, independent of class grouping and legacy interleaving.
    trials.sort(key=lambda trial: (trial.session_id, trial.trial_id))
    by_session = {session.session_id: session for session in sessions}
    identities = {(trial.session_id, trial.trial_id) for trial in trials}
    exclusions = []
    for session in sessions:
        if session.session_id not in config.trials.session_indices:
            continue
        for index, label in enumerate(session.labels):
            if (session.session_id, index) not in identities:
                exclusions.append(dict(subject_id=subject, session_id=session.session_id,
                                       original_trial_index=index, original_class_label=int(label),
                                       reason='unselected_class' if int(label) not in dict(config.trials.label_map)
                                       else 'incomplete_trial'))
    rows = []
    for i, trial in enumerate(trials):
        session = by_session[trial.session_id]
        position = session.positions[trial.trial_id].item()
        rows.append(dict(source_file=Path(path).name, subject_id=subject,
                         session_id=trial.session_id, original_trial_index=trial.trial_id,
                         original_event_position=position,
                         start_sample=int(position) + config.trials.event_index_offset,
                         original_class_label=int(session.labels[trial.trial_id]), class_label=trial.label,
                         array_index=i, split_seed=config.seed))
    if not trials:
        raise ValueError(f'Subject {subject} has no eligible trials')
    return np.stack([trial.signal for trial in trials]), rows, exclusions


def validate_config(config):
    if not config.subjects or len(set(config.subjects)) != len(config.subjects) or any(s < 1 for s in config.subjects):
        raise ValueError('Subjects must be unique positive IDs')
    if config.seed < 0 or config.seed >= 2**32:
        raise ValueError('Seed must be in [0, 2**32)')
    if len(config.percentages) != 3 or any(type(p) is not int or p <= 0 for p in config.percentages) or sum(config.percentages) != 100:
        raise ValueError('Three positive integer percentages must sum to 100')
    if config.sample_rate <= 0 or config.trials.duration_seconds <= 0:
        raise ValueError('Sample rate and trial duration must be positive')
    for values in (config.trials.session_indices, config.trials.channel_indices):
        if not values or len(set(values)) != len(values) or min(values) < 0:
            raise ValueError('Session/channel indices must be unique and nonnegative')
    if config.trials.label_map != ((1, 0), (2, 1)):
        raise ValueError('Original split expects left/right labels 1/2 mapped to 0/1')


def verify_saved(output, expected: SplitMetadata | None = None) -> SplitMetadata:
    output = Path(output)
    metadata: SplitMetadata = json.loads((output / 'metadata.json').read_text())
    if expected is not None:
        for key in ('config', 'source_sha256', 'format_version'):
            if metadata[key] != expected[key]:
                raise ValueError(f'Existing split has different {key}; use a new output directory')
    for name, digest in metadata['artifact_sha256'].items():
        if sha256(output / name) != digest:
            raise ValueError(f'Saved artifact changed: {name}')
    return metadata


def load_partition(output, subject_id, partition):
    """Load saved EEG arrays in saved partition order, without resplitting."""
    if partition not in PARTITIONS:
        raise ValueError(f'Partition must be one of {PARTITIONS}')
    verify_saved(output)
    with (Path(output) / 'partitions.csv').open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row['subject_id']) == subject_id and row['partition'] == partition]
    if not rows:
        raise ValueError(f'No saved {partition} trials for subject {subject_id}')
    rows.sort(key=lambda row: int(row['partition_index']))
    indices = [int(row['array_index']) for row in rows]
    with np.load(Path(output) / f'subject_{subject_id:02d}.npz', allow_pickle=False) as data:
        return data['signals'][indices], data['labels'][indices], rows


def create_original_split(data_dir, output=DEFAULT_OUTPUT, config=SplitConfig()) -> SplitMetadata:
    validate_config(config)
    output = Path(output).resolve()
    paths = {s: subject_file_path(data_dir, s) for s in config.subjects}
    metadata = SplitMetadata(format_version=1, config=json.loads(json.dumps(asdict(config))),
                    source_sha256={p.name: sha256(p) for p in paths.values()},
                    source_directory=str(Path(data_dir).resolve()), numpy_version=np.__version__,
                    signal_axes=['trial', 'sample', 'channel'], processing='original trial extraction only',
                    rounding='largest remainder; ties: training, validation, testing',
                    randomization='PCG64 SeedSequence([seed, subject_id, stage, class_or_partition_index]); stage 0=class, 1=partition',
                    index_convention='zero-based session, original event index, array index, partition index',
                    counts={}, exclusions=[], artifact_sha256={})
    if output.exists() and any(output.iterdir()):
        saved = verify_saved(output, metadata)
        print(f'Reused verified partitions: {output}')
        return saved
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.original-', dir=output.parent))
    try:
        all_rows = []
        for subject, path in paths.items():
            signals, rows, exclusions = extract_original(path, subject, config)
            labels = np.array([row['class_label'] for row in rows], dtype=np.int64)
            parts = split_indices(labels, subject, config)
            flat = [i for indices in parts.values() for i in indices]
            if sorted(flat) != list(range(len(rows))):
                raise ValueError('Partitions must cover every eligible trial exactly once')
            metadata['counts'][str(subject)] = {}
            for name, indices in parts.items():
                metadata['counts'][str(subject)][name] = {str(label): int(sum(labels[indices] == label)) for label in (0, 1)}
                for position, index in enumerate(indices):
                    all_rows.append(dict(rows[index], partition=name, partition_index=position))
            metadata['exclusions'].extend(exclusions)
            np.savez_compressed(staging / f'subject_{subject:02d}.npz', signals=signals, labels=labels,
                                subject_ids=np.full(len(rows), subject, dtype=np.int64),
                                session_ids=np.array([row['session_id'] for row in rows]),
                                original_trial_indices=np.array([row['original_trial_index'] for row in rows]))
        with (staging / 'partitions.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        metadata['artifact_sha256'] = {p.name: sha256(p) for p in sorted(staging.iterdir())}
        (staging / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if output.exists():
            output.rmdir()  # Only an empty destination can be replaced.
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f'Saved original EEG and partitions: {output}')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--subjects', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    config = SplitConfig(seed=args.seed)
    if args.subjects is not None:
        config = replace(config, subjects=tuple(args.subjects))
    metadata = create_original_split(resolve_data_dir(args.data_dir), args.output, config)
    for subject, counts in metadata['counts'].items():
        print(f'Subject {subject}: {counts}')


if __name__ == '__main__':
    main()
