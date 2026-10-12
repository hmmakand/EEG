"""Load finalized LOSO assignments using the existing Version18 graph builder."""
from collections import defaultdict
import json

from .dataset import (SavedWithinSubject, FoldGraphs, default_dataset_config,
                      validate_config, verify_bundle, verify_hash, read_rows, sha256)
from ..datasetsynthetic.config import PreparationConfig
from ..datasetsynthetic.protocols import validate_plans
from ..datasetsynthetic.generate import verify_generation


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

    def _load_generator(self, job_id):
        job = self.jobs[job_id]
        subject = job['subject_id']
        name = f'{job["run_directory"]}/subject_{subject:02d}.npz'
        if name in self._source_hashes:
            return
        if not verify_generation(self.root, job):
            raise ValueError(f'Incomplete generator: {job_id}')
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
