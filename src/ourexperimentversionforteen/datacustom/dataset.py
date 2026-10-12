"""Build graph loaders from finalized within-subject signal assignments; never split."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

if __package__ == 'datacustom':
    from dataset import DatasetConfig, default_dataset_config, build_trial_graph
    from dataset.trials import Trial
    from dataset.validation import validate_config
    from datasetsynthetic.storage import read_rows, sha256, verify_bundle
else:
    from ..dataset import DatasetConfig, default_dataset_config, build_trial_graph
    from ..dataset.trials import Trial
    from ..dataset.validation import validate_config
    from ..datasetsynthetic.storage import read_rows, sha256, verify_bundle

BUNDLE = Path(__file__).resolve().parents[1] / 'datasyn' / 'fold_first'


def verify_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'Saved artifact changed: {path}')
    return actual


def counts(graphs):
    return dict(total=len(graphs), original=sum(not bool(g.is_synthetic) for g in graphs),
                synthetic=sum(bool(g.is_synthetic) for g in graphs),
                class_counts={str(c): sum(int(g.y.item()) == c for g in graphs) for c in (0, 1)})


@dataclass
class FoldGraphs:
    subject: int
    fold: int
    training: list[Data]
    testing: list[Data]
    source_sha256: dict[str, str]

    def loaders(self, batch_size: int, train_shuffle: bool = False):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        return (DataLoader(self.training, batch_size=batch_size, shuffle=train_shuffle),
                DataLoader(self.testing, batch_size=batch_size, shuffle=False))


class SavedWithinSubject:
    """Read one completed protocol even when cross-subject generation is pending.

    Original graphs are cached for the current subject only. Synthetic graphs
    are built per requested fold and are owned by the returned FoldGraphs.
    """
    def __init__(self, bundle: str | Path = BUNDLE, dataset_config: DatasetConfig | None = None):
        self.root = Path(bundle).resolve()
        self.config = dataset_config or default_dataset_config()
        validate_config(self.config)
        self.metadata = verify_bundle(self.root)
        status = json.loads((self.root / 'generation_status.json').read_text())['protocols']['within_subject']
        if not status['complete']:
            raise ValueError('Within-subject generation is not complete; finalize it before graph training')
        self.manifest_path = self.root / 'protocols' / 'within_subject.csv'
        self.manifest_sha256 = verify_hash(self.manifest_path, status['manifest_sha256'])
        if self.manifest_sha256 != self.metadata['artifact_sha256']['protocols/within_subject.plan.csv']:
            raise ValueError('Finalized fold assignments differ from the prepared plan')
        extraction = self.metadata['config']
        if extraction['sample_rate'] != self.config.sample_rate:
            raise ValueError('Graph sample rate differs from signal extraction')
        for name in ('channel_indices', 'session_indices', 'label_map', 'duration_seconds', 'event_index_offset'):
            if extraction['trials'][name] != json.loads(json.dumps(getattr(self.config.trials, name))):
                raise ValueError(f'Graph configuration differs from signal extraction: {name}')
        self.subjects = tuple(extraction['subjects'])
        originals = read_rows(self.root / 'original' / 'trials.csv')
        self.originals = {r['trial_id']: r for r in originals}
        if len(self.originals) != len(originals):
            raise ValueError('Duplicate original trial identities')
        jobs = json.loads((self.root / 'jobs.json').read_text())
        self.jobs = {j['job_id']: j for j in jobs}
        self.assignments = defaultdict(list)
        for row in read_rows(self.manifest_path):
            self.assignments[(int(row['target_subject']), int(row['fold']))].append(row)
        self._validate_assignments(extraction['n_folds'])
        self._cached_subject = None
        self._original_graphs: dict[str, Data] = {}

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Signal reference escapes the prepared bundle')
        return path

    def _validate_assignments(self, n_folds):
        expected = {(s, f) for s in self.subjects for f in range(1, n_folds + 1)}
        if set(self.assignments) != expected:
            raise ValueError('Saved subjects/folds differ from preparation metadata')
        tested = Counter()
        for (subject, fold), rows in self.assignments.items():
            original_ids = {key for key, r in self.originals.items() if int(r['subject_id']) == subject}
            ids = [r['trial_id'] for r in rows]
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate or overlapping trial identities in fold')
            if any(r['protocol'] != 'within_subject' or int(r['subject_id']) != subject or
                   r['role'] not in ('training', 'testing') or r['source'] not in ('original', 'synthetic') for r in rows):
                raise ValueError('Invalid protocol, subject, role, or source in fold')
            test = [r for r in rows if r['role'] == 'testing']
            real = [r for r in rows if r['source'] == 'original']
            synthetic = [r for r in rows if r['source'] == 'synthetic']
            if not test or not synthetic or any(r['source'] != 'original' for r in test):
                raise ValueError('Synthetic trials must be training-only; original testing must be nonempty')
            if {r['trial_id'] for r in real} != original_ids:
                raise ValueError('Fold must contain each subject original exactly once')
            for row in real:
                source = self.originals[row['trial_id']]
                if row['generator_job'] or any(row[k] != source[k] for k in ('signal_file', 'array_index', 'class_label', 'subject_id')):
                    raise ValueError('Original assignment differs from saved trial identity')
            job_id = f'within_subject/S{subject:02d}/fold_{fold:02d}'
            job = self.jobs[job_id]
            permitted = {r['trial_id'] for r in real if r['role'] == 'training'}
            if (job['protocol'] != 'within_subject' or job['subject_id'] != subject or
                    set(job['original_trial_ids']) != permitted):
                raise ValueError('Diffusion inputs differ from classifier training originals')
            if len(synthetic) != job['synthetic_trial_count']:
                raise ValueError('Synthetic count differs from the prepared job')
            if sorted(int(r['array_index']) for r in synthetic) != list(range(len(synthetic))):
                raise ValueError('Synthetic indices must cover each generated trial once')
            for r in synthetic:
                i = int(r['array_index'])
                if (r['generator_job'] != job_id or r['trial_id'] != f'synthetic:{job_id}:{i:05d}' or
                        r['signal_file'] != f'{job["run_directory"]}/subject_{subject:02d}.npz' or
                        int(r['class_label']) != i // self.metadata['config']['samples_per_class']):
                    raise ValueError('Synthetic assignment belongs to a different fold or generation row')
            tested.update(r['trial_id'] for r in test)
        if set(tested) != set(self.originals) or any(n != 1 for n in tested.values()):
            raise ValueError('Each original must be tested exactly once across its saved folds')

    def fold_numbers(self, subject):
        if subject not in self.subjects:
            raise ValueError(f'Subject {subject} is not in the saved protocol')
        return sorted(fold for s, fold in self.assignments if s == subject)

    def _graphs(self, path, rows, *, synthetic=False, generation_rows=None):
        graphs = {}
        with np.load(path, allow_pickle=False) as data:
            signals, labels = data['signals'], data['labels']
            shape = (int(self.config.sample_rate * self.config.trials.duration_seconds), len(self.config.trials.channel_indices))
            if (signals.shape != (len(rows), *shape) or labels.shape != (len(rows),) or
                    not np.isfinite(signals).all() or not np.issubdtype(labels.dtype, np.integer) or
                    not set(labels.tolist()) <= {0, 1}):
                raise ValueError('Unexpected signal dimensions, values, or labels')
            if sorted(int(r['array_index']) for r in rows) != list(range(len(rows))):
                raise ValueError('Signal rows are not covered exactly once')
            for r in rows:
                index, subject = int(r['array_index']), int(r['subject_id'])
                if int(labels[index]) != int(r['class_label']) or int(data['subject_ids'][index]) != subject:
                    raise ValueError('Array label/subject differs from saved assignment')
                if synthetic:
                    assert generation_rows is not None
                    source = generation_rows[index]
                    if str(data['synthetic_trial_ids'][index]) != source['synthetic_trial_id']:
                        raise ValueError('Synthetic array identity differs from generation manifest')
                    session, trial = -1, index
                else:
                    session, trial = int(r['session_id']), int(r['original_trial_index'])
                    if (int(data['session_ids'][index]) != session or int(data['original_trial_indices'][index]) != trial or
                            str(data['trial_ids'][index]) != r['trial_id']):
                        raise ValueError('Original array identity differs from trial manifest')
                graph = build_trial_graph(Trial(signals[index], int(labels[index]), subject, session, trial), self.config)
                graph.sample_id = r['trial_id']
                graph.is_synthetic = synthetic
                graph.generator_job = r['generator_job'] if synthetic else ''
                graphs[r['trial_id']] = graph
        return graphs

    def load_fold(self, subject: int, fold: int) -> FoldGraphs:
        if fold not in self.fold_numbers(subject):
            raise ValueError(f'Fold {fold} is not saved for subject {subject}')
        original_name = f'original/subject_{subject:02d}.npz'
        original_hash = self.metadata['artifact_sha256'][original_name]
        if self._cached_subject != subject:
            verify_hash(self.path(original_name), original_hash)
            rows = [r for r in self.originals.values() if int(r['subject_id']) == subject]
            self._original_graphs = self._graphs(self.path(original_name), rows)
            self._cached_subject = subject
        assignments = self.assignments[(subject, fold)]
        synthetic = [r for r in assignments if r['source'] == 'synthetic']
        job = self.jobs[synthetic[0]['generator_job']]
        run = self.path(job['run_directory'])
        meta_path = run / 'metadata.json'
        meta = json.loads(meta_path.read_text())
        if meta['status'] != 'complete':
            raise ValueError('Fold generation is not complete')
        generation = meta['generation'][str(subject)]
        config = json.loads((run / 'config.json').read_text())
        if (config['subjects'] != [subject] or config['samples_per_class'] * 2 != len(synthetic) or
                config['validation_per_class'] != self.metadata['config']['validation_per_class'] or
                meta['subjects'][str(subject)]['epoch'] != config['epochs']):
            raise ValueError('Generator configuration differs from completed fold')
        expected_sources = {name: self.metadata['artifact_sha256'][f'{job["input_directory"]}/{name}']
                            for name in ('metadata.json', 'partitions.csv')}
        expected_sources[f'subject_{subject:02d}.npz'] = original_hash
        if meta['original_sha256'] != expected_sources:
            raise ValueError('Synthetic run used different permitted original inputs')
        internal_hash = verify_hash(run / 'internal_partitions.csv', meta['internal_partitions_sha256'])
        internal = read_rows(run / 'internal_partitions.csv')
        if (len(internal) != len(job['original_trial_ids']) or
                {r['trial_id'] for r in internal} != set(job['original_trial_ids']) or
                Counter(r['diffusion_subset'] for r in internal) !=
                {'training': job['training_trial_count'], 'validation': job['validation_trial_count']}):
            raise ValueError('Diffusion fitting/validation rows differ from permitted fold originals')
        for r in internal:
            original = self.originals[r['trial_id']]
            if any(r[k] != original[k] for k in ('array_index', 'class_label', 'subject_id', 'session_id', 'original_trial_index')):
                raise ValueError('Diffusion internal identity differs from original trial')
        manifest_hash = verify_hash(run / 'manifest.csv', meta['manifest_sha256'])
        generated_rows = read_rows(run / 'manifest.csv')
        generated_rows.sort(key=lambda r: int(r['array_index']))
        if len(generated_rows) != len(synthetic):
            raise ValueError('Generation manifest size differs from saved fold')
        for i, r in enumerate(generated_rows):
            if (int(r['array_index']) != i or int(r['subject_id']) != subject or
                    int(r['class_label']) != i // config['samples_per_class'] or
                    r['output_file'] != f'subject_{subject:02d}.npz' or
                    r['checkpoint_sha256'] != generation['checkpoint_sha256'] or
                    r['source'] != 'synthetic' or r['intended_partition'] != 'training'):
                raise ValueError('Generation manifest identity/provenance mismatch')
        signal_path = self.path(synthetic[0]['signal_file'])
        signal_hash = verify_hash(signal_path, generation['sha256'])
        graphs = self._graphs(signal_path, synthetic, synthetic=True, generation_rows=generated_rows)
        graphs.update(self._original_graphs)
        return FoldGraphs(subject, fold,
                          [graphs[r['trial_id']] for r in assignments if r['role'] == 'training'],
                          [graphs[r['trial_id']] for r in assignments if r['role'] == 'testing'],
                          {original_name: original_hash, synthetic[0]['signal_file']: signal_hash,
                           f'{job["run_directory"]}/manifest.csv': manifest_hash,
                           f'{job["run_directory"]}/internal_partitions.csv': internal_hash,
                           f'{job["run_directory"]}/metadata.json': sha256(meta_path),
                           f'{job["run_directory"]}/config.json': sha256(run / 'config.json')})
