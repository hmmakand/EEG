"""Load finalized LOSO assignments using the existing Version17 graph builder."""
from collections import defaultdict, Counter
import json

from .dataset import (SavedWithinSubject, FoldGraphs, default_dataset_config,
                      validate_config, verify_bundle, verify_hash, read_rows, sha256)
from ..datasetsynthetic.config import PreparationConfig
from ..datasetsynthetic.protocols import validate_plans


class SavedLOSO(SavedWithinSubject):
    """Reuse signal/graph conversion; replace within-subject protocol loading."""

    def __init__(self, bundle, dataset_config=None):
        from pathlib import Path
        self.root = Path(bundle).resolve()
        self.config = dataset_config or default_dataset_config()
        validate_config(self.config)
        self.metadata = verify_bundle(self.root)
        status = json.loads((self.root / 'generation_status.json').read_text())['protocols']['cross_subject']
        if not status['complete']:
            raise ValueError('Cross-subject generation is not complete')
        self.manifest_path = self.root / 'protocols/cross_subject.csv'
        self.manifest_sha256 = verify_hash(self.manifest_path, status['manifest_sha256'])
        if self.manifest_sha256 != self.metadata['artifact_sha256']['protocols/cross_subject.plan.csv']:
            raise ValueError('Finalized LOSO assignments differ from prepared plan')
        extraction = self.metadata['config']
        if extraction['sample_rate'] != self.config.sample_rate:
            raise ValueError('Graph sample rate differs from signal extraction')
        for name in ('channel_indices', 'session_indices', 'label_map', 'duration_seconds', 'event_index_offset'):
            if extraction['trials'][name] != json.loads(json.dumps(getattr(self.config.trials, name))):
                raise ValueError(f'Graph configuration differs from signal extraction: {name}')
        self.subjects = tuple(extraction['subjects'])
        originals = read_rows(self.root / 'original/trials.csv')
        self.originals = {r['trial_id']: r for r in originals}
        jobs = json.loads((self.root / 'jobs.json').read_text())
        self.jobs = {j['job_id']: j for j in jobs}
        rows = read_rows(self.manifest_path)
        validate_plans({'cross_subject': rows}, jobs, originals,
                       PreparationConfig(subjects=self.subjects))
        self.assignments = defaultdict(list)
        for row in rows:
            self.path(row['signal_file'])
            if (row['protocol'] != 'cross_subject' or row['role'] not in ('training', 'testing')
                    or row['source'] not in ('original', 'synthetic')):
                raise ValueError('Expected cross_subject assignments')
            self.assignments[(int(row['target_subject']), int(row['fold']))].append(row)
            if row['source'] == 'original':
                original = self.originals[row['trial_id']]
                if row['generator_job'] or any(row[k] != original[k] for k in ('signal_file', 'array_index', 'class_label', 'subject_id')):
                    raise ValueError('Original assignment identity mismatch')
            else:
                job = self.jobs[row['generator_job']]
                subject, index = int(row['subject_id']), int(row['array_index'])
                if (job['protocol'] != 'cross_subject' or job['subject_id'] != subject or
                        row['trial_id'] != f'synthetic:{job["job_id"]}:{index:05d}' or
                        row['signal_file'] != f'{job["run_directory"]}/subject_{subject:02d}.npz' or
                        not 0 <= index < job['synthetic_trial_count'] or
                        int(row['class_label']) != index // extraction['samples_per_class']):
                    raise ValueError('Synthetic assignment identity mismatch')
        if ({subject for subject, _ in self.assignments} != set(self.subjects) or
                any(len(self.fold_numbers(subject)) != 1 for subject in self.subjects)):
            raise ValueError('Expected one LOSO fold per saved subject')
        self._graphs_by_id = {}
        self._source_hashes = {}

    def _load_subject(self, subject):
        name = f'original/subject_{subject:02d}.npz'
        if name in self._source_hashes:
            return
        self._source_hashes[name] = verify_hash(self.path(name), self.metadata['artifact_sha256'][name])
        rows = [r for r in self.originals.values() if int(r['subject_id']) == subject]
        self._graphs_by_id.update(self._graphs(self.path(name), rows))

    def _verify_generator(self, job):
        """Validate saved artifacts independently of generator architecture settings."""
        subject = job['subject_id']
        run = self.path(job['run_directory'])
        meta = json.loads((run / 'metadata.json').read_text())
        config = json.loads((run / 'config.json').read_text())
        if (meta['status'] != 'complete' or config['subjects'] != [subject] or
                config['samples_per_class'] * 2 != job['synthetic_trial_count'] or
                config['validation_per_class'] != self.metadata['config']['validation_per_class'] or
                meta['subjects'][str(subject)]['epoch'] != config['epochs']):
            raise ValueError('Generator configuration differs from completed job')
        expected = {name: self.metadata['artifact_sha256'][f'{job["input_directory"]}/{name}']
                    for name in ('metadata.json', 'partitions.csv')}
        expected[f'subject_{subject:02d}.npz'] = self.metadata['artifact_sha256'][f'original/subject_{subject:02d}.npz']
        if meta['original_sha256'] != expected:
            raise ValueError('Generator used different permitted originals')
        verify_hash(run / 'internal_partitions.csv', meta['internal_partitions_sha256'])
        internal = read_rows(run / 'internal_partitions.csv')
        if (len(internal) != len(job['original_trial_ids']) or
                {r['trial_id'] for r in internal} != set(job['original_trial_ids']) or
                Counter(r['diffusion_subset'] for r in internal) !=
                {'training': job['training_trial_count'], 'validation': job['validation_trial_count']}):
            raise ValueError('Generator internal partitions differ from permitted originals')
        for row in internal:
            original = self.originals[row['trial_id']]
            if any(row[k] != original[k] for k in ('array_index', 'class_label', 'subject_id', 'session_id', 'original_trial_index')):
                raise ValueError('Generator internal trial identity mismatch')
        generation = meta['generation'][str(subject)]
        verify_hash(run / f'subject_{subject:02d}.npz', generation['sha256'])
        verify_hash(run / 'manifest.csv', meta['manifest_sha256'])
        verify_hash(run / 'checkpoints' / f'subject_{subject:02d}' / 'best.pt', generation['checkpoint_sha256'])
        generated = sorted(read_rows(run / 'manifest.csv'), key=lambda r: int(r['array_index']))
        if len(generated) != job['synthetic_trial_count']:
            raise ValueError('Generated manifest count differs from prepared job')
        for i, row in enumerate(generated):
            if (int(row['array_index']) != i or int(row['subject_id']) != subject or
                    int(row['class_label']) != i // config['samples_per_class'] or
                    row['output_file'] != f'subject_{subject:02d}.npz' or
                    row['checkpoint_sha256'] != generation['checkpoint_sha256'] or
                    row['source'] != 'synthetic' or row['intended_partition'] != 'training'):
                raise ValueError('Generated manifest identity/provenance mismatch')

    def _load_generator(self, job_id):
        job = self.jobs[job_id]
        subject = job['subject_id']
        name = f'{job["run_directory"]}/subject_{subject:02d}.npz'
        if name in self._source_hashes:
            return
        self._verify_generator(job)
        rows = next([r for r in assignments if r['generator_job'] == job_id]
                    for assignments in self.assignments.values()
                    if any(r['generator_job'] == job_id for r in assignments))
        if sorted(int(r['array_index']) for r in rows) != list(range(job['synthetic_trial_count'])):
            raise ValueError('Synthetic rows do not cover the prepared job')
        run = self.path(job['run_directory'])
        generated = sorted(read_rows(run / 'manifest.csv'), key=lambda r: int(r['array_index']))
        self._graphs_by_id.update(self._graphs(self.path(name), rows, synthetic=True, generation_rows=generated))
        for suffix in (f'subject_{subject:02d}.npz', 'metadata.json', 'manifest.csv', 'internal_partitions.csv', 'config.json'):
            relative = f'{job["run_directory"]}/{suffix}'
            self._source_hashes[relative] = sha256(self.path(relative))

    def load_fold(self, subject, fold):
        if fold not in self.fold_numbers(subject):
            raise ValueError(f'Fold {fold} is not saved for subject {subject}')
        rows = self.assignments[(subject, fold)]
        for source in sorted({int(r['subject_id']) for r in rows}):
            self._load_subject(source)
        for job_id in sorted({r['generator_job'] for r in rows if r['source'] == 'synthetic'}):
            self._load_generator(job_id)
        names = {r['signal_file'] for r in rows}
        for r in rows:
            if r['source'] == 'synthetic':
                run = self.jobs[r['generator_job']]['run_directory']
                names.update(f'{run}/{suffix}' for suffix in ('metadata.json', 'manifest.csv', 'internal_partitions.csv', 'config.json'))
        return FoldGraphs(subject, fold,
                          [self._graphs_by_id[r['trial_id']] for r in rows if r['role'] == 'training'],
                          [self._graphs_by_id[r['trial_id']] for r in rows if r['role'] == 'testing'],
                          {name: self._source_hashes[name] for name in sorted(names)})
