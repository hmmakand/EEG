"""Persist original-only folds and permitted diffusion inputs before generation."""
import os
from collections import defaultdict
from sklearn.model_selection import KFold
from .storage import write_json, write_rows, sha256

PROTOCOLS = ('within_subject', 'cross_subject')


def assignment(protocol, fold, target, role, row, *, job=''):
    return dict(protocol=protocol, fold=str(fold), target_subject=str(target),
                role=role, subject_id=str(row['subject_id']), trial_id=row['trial_id'],
                source='synthetic' if job else 'original', class_label=str(row['class_label']),
                signal_file=row['signal_file'], array_index=str(row['array_index']), generator_job=job)


def make_job(root, job_id, protocol, subject, rows, config):
    for label in (0, 1):
        if sum(int(r['class_label']) == label for r in rows) <= config.validation_per_class:
            raise ValueError(f'{job_id}: too few permitted class-{label} trials for diffusion validation')
    path = root / 'inputs' / job_id
    path.mkdir(parents=True)
    selected = [dict(r, partition='training', partition_index=i, split_seed=config.seed)
                for i, r in enumerate(rows)]
    write_rows(path / 'partitions.csv', selected)
    name = f'subject_{subject:02d}.npz'
    write_json(path / 'metadata.json', dict(
        format_version=2, purpose='permitted diffusion inputs; not classifier partitions',
        job_id=job_id, protocol=protocol,
        signal_root=os.path.relpath(root / 'original', path),
        artifact_sha256={'partitions.csv': sha256(path / 'partitions.csv'),
                         name: sha256(root / 'original' / name)},
        original_trial_ids=[r['trial_id'] for r in rows]))
    return dict(job_id=job_id, protocol=protocol, subject_id=subject,
                input_directory=f'inputs/{job_id}', run_directory=f'synthetic/{job_id}',
                original_trial_ids=[r['trial_id'] for r in rows],
                training_trial_count=len(rows) - 2 * config.validation_per_class,
                validation_trial_count=2 * config.validation_per_class,
                synthetic_trial_count=2 * config.samples_per_class)


def synthetic_rows(job, config):
    for i in range(2 * config.samples_per_class):
        yield dict(subject_id=job['subject_id'], trial_id=f'synthetic:{job["job_id"]}:{i:05d}',
                   class_label=i // config.samples_per_class,
                   signal_file=f'{job["run_directory"]}/subject_{job["subject_id"]:02d}.npz', array_index=i)


def save_protocols(root, rows, config):
    by_subject = {s: sorted((r for r in rows if int(r['subject_id']) == s),
                           key=lambda r: (int(r['session_id']), int(r['original_trial_index'])))
                  for s in config.subjects}
    plans = {p: [] for p in PROTOCOLS}
    jobs = []
    for subject, originals in by_subject.items():
        splitter = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
        for fold, (train, test) in enumerate(splitter.split(originals), 1):
            job = make_job(root, f'within_subject/S{subject:02d}/fold_{fold:02d}',
                           'within_subject', subject, [originals[i] for i in train], config)
            jobs.append(job)
            for role, indices in [('training', train), ('testing', test)]:
                plans['within_subject'].extend(assignment('within_subject', fold, subject, role, originals[i])
                                               for i in indices)
            plans['within_subject'].extend(assignment('within_subject', fold, subject, 'training', r,
                                                     job=job['job_id']) for r in synthetic_rows(job, config))
    cross_jobs = {}
    for subject, originals in by_subject.items():
        job = make_job(root, f'cross_subject/S{subject:02d}', 'cross_subject', subject, originals, config)
        jobs.append(job)
        cross_jobs[subject] = job
    for fold, held_out in enumerate(config.subjects, 1):
        for subject, originals in by_subject.items():
            role = 'testing' if subject == held_out else 'training'
            plans['cross_subject'].extend(assignment('cross_subject', fold, held_out, role, r) for r in originals)
            if subject != held_out:
                job = cross_jobs[subject]
                plans['cross_subject'].extend(assignment('cross_subject', fold, held_out, role, r, job=job['job_id'])
                                             for r in synthetic_rows(job, config))
    validate_plans(plans, jobs, rows, config)
    for protocol, plan in plans.items():
        write_rows(root / 'protocols' / f'{protocol}.plan.csv', plan)
    write_json(root / 'jobs.json', jobs)
    return jobs


def validate_plans(plans, jobs, originals, config):
    """Check classifier disjointness AND all generator fitting/selection inputs."""
    original_ids = {r['trial_id'] for r in originals}
    job_sources = {j['job_id']: set(j['original_trial_ids']) for j in jobs}
    if len(original_ids) != len(originals) or len(job_sources) != len(jobs):
        raise ValueError('Duplicate original or generator identities')
    for protocol, plan in plans.items():
        grouped = defaultdict(list)
        for r in plan:
            grouped[(r['target_subject'], r['fold'])].append(r)
        expected_folds = len(config.subjects) * (config.n_folds if protocol == 'within_subject' else 1)
        if len(grouped) != expected_folds:
            raise ValueError('Unexpected number of classifier folds')
        tested = []
        for (target, _), assignments in grouped.items():
            train = [r for r in assignments if r['role'] == 'training']
            test = [r for r in assignments if r['role'] == 'testing']
            train_ids, test_ids = {r['trial_id'] for r in train}, {r['trial_id'] for r in test}
            if not train or not test or len(train_ids) + len(test_ids) != len(assignments) or train_ids & test_ids:
                raise ValueError('Missing or overlapping classifier assignments')
            if any(r['source'] != 'original' for r in test):
                raise ValueError('Test data must be original')
            permitted = {r['trial_id'] for r in train if r['source'] == 'original'}
            for r in train:
                if r['source'] == 'synthetic' and not job_sources[r['generator_job']].issubset(permitted):
                    raise ValueError('Generator inputs include trials outside classifier training')
            if protocol == 'within_subject':
                if any(int(r['subject_id']) != int(target) for r in assignments):
                    raise ValueError('Within-subject fold contains another subject')
            elif any(int(r['subject_id']) == int(target) for r in train) or any(int(r['subject_id']) != int(target) for r in test):
                raise ValueError('Held-out subject appears in training')
            tested.extend(test_ids)
        if len(tested) != len(original_ids) or set(tested) != original_ids:
            raise ValueError('Each original must be tested exactly once per protocol')
