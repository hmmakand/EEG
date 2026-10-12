"""Train prepared diffusion jobs and finalize manifests only after verified generation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'datasetsynthetic'

from .config import DEFAULT_OUTPUT
from .protocols import PROTOCOLS
from .storage import read_rows, sha256, verify_bundle, write_json, write_rows


def verify_generation(root, job):
    """Validate a complete job before publishing references to its signal rows."""
    from .eegdiffuser.data import original_fingerprints, validate_internal_partitions
    from .eegdiffuser.runtime import load_config
    run = root / job['run_directory']
    if not (run / 'metadata.json').exists():
        return False
    metadata = json.loads((run / 'metadata.json').read_text())
    if metadata['status'] != 'complete':
        return False
    config = load_config(run)
    subject = job['subject_id']
    if config.subjects != (subject,) or 2 * config.samples_per_class != job['synthetic_trial_count']:
        raise ValueError(f'Generation configuration differs from prepared job: {job["job_id"]}')
    original = root / job['input_directory']
    if metadata['original_sha256'] != original_fingerprints(original, config):
        raise ValueError('Generation used different permitted originals')
    validate_internal_partitions(original, run / 'internal_partitions.csv', config)
    if sha256(run / 'internal_partitions.csv') != metadata['internal_partitions_sha256']:
        raise ValueError('Diffusion internal partitions changed')
    generated = metadata['generation'][str(subject)]
    name = f'subject_{subject:02d}.npz'
    if sha256(run / name) != generated['sha256'] or sha256(run / 'manifest.csv') != metadata['manifest_sha256']:
        raise ValueError('Generated signals or manifest changed')
    best = run / 'checkpoints' / f'subject_{subject:02d}' / 'best.pt'
    if sha256(best) != generated['checkpoint_sha256']:
        raise ValueError('Generation checkpoint changed')
    rows = read_rows(run / 'manifest.csv')
    if len(rows) != job['synthetic_trial_count']:
        raise ValueError('Synthetic manifest count differs from prepared plan')
    with np.load(run / name, allow_pickle=False) as arrays:
        labels = np.repeat([0, 1], config.samples_per_class)
        if arrays['signals'].shape != (len(labels), config.time_points, config.in_channels) or not np.isfinite(arrays['signals']).all():
            raise ValueError('Unexpected generated signal shape or non-finite values')
        if not np.array_equal(arrays['labels'], labels) or not np.all(arrays['subject_ids'] == subject):
            raise ValueError('Generated labels or subject IDs differ from plan')
        for i, row in enumerate(rows):
            if (int(row['array_index']) != i or int(row['class_label']) != int(labels[i]) or
                int(row['subject_id']) != subject or row['output_file'] != name or
                row['synthetic_trial_id'] != arrays['synthetic_trial_ids'][i] or
                row['checkpoint_sha256'] != generated['checkpoint_sha256']):
                raise ValueError('Generated manifest identity/provenance mismatch')
    return True


def finalize(root):
    root = Path(root).resolve()
    metadata = verify_bundle(root)
    jobs = json.loads((root / 'jobs.json').read_text())
    complete = {job['job_id']: verify_generation(root, job) for job in jobs}
    status = {}
    for protocol in PROTOCOLS:
        relevant = [j for j in jobs if j['protocol'] == protocol]
        ready = all(complete[j['job_id']] for j in relevant)
        status[protocol] = dict(complete=ready, jobs_complete=sum(complete[j['job_id']] for j in relevant), jobs_total=len(relevant))
        if ready:
            rows = read_rows(root / 'protocols' / f'{protocol}.plan.csv')
            temporary = root / 'protocols' / f'{protocol}.csv.tmp'
            write_rows(temporary, rows)
            final = root / 'protocols' / f'{protocol}.csv'
            temporary.replace(final)
            status[protocol]['manifest_sha256'] = sha256(final)
    finished = all(item['complete'] for item in status.values())
    write_json(root / 'generation_status.json', dict(complete=finished, protocols=status))
    metadata.update(status='complete' if finished else 'prepared_or_partially_generated', synthetic_generation_complete=finished)
    write_json(root / 'metadata.json', metadata)
    return status


def generate(root=DEFAULT_OUTPUT, *, protocols=PROTOCOLS, subjects=None, folds=None,
             epochs=None, device=None, resume=False, diffusion_config=None):
    from .eegdiffuser.config import Config
    from .eegdiffuser.runtime import load_config
    from .eegdiffuser.train import run_training
    root = Path(root).resolve()
    metadata = verify_bundle(root)
    settings = metadata['config']
    config = diffusion_config or Config()
    overrides = dict(samples_per_class=settings['samples_per_class'], validation_per_class=settings['validation_per_class'],
                     split_seed=settings['seed'], time_points=int(settings['sample_rate'] * settings['trials']['duration_seconds']),
                     in_channels=len(settings['trials']['channel_indices']))
    if epochs is not None:
        overrides['epochs'] = epochs
    if device is not None:
        overrides['device'] = device
    config = replace(config, **overrides)
    config.validate()
    jobs = json.loads((root / 'jobs.json').read_text())
    if not protocols or not set(protocols).issubset(PROTOCOLS):
        raise ValueError('Unknown or empty protocol selection')
    if folds and 'cross_subject' in protocols:
        raise ValueError('--folds selects within-subject generator jobs only; choose --protocol within_subject')
    if subjects and not set(subjects).issubset(settings['subjects']):
        raise ValueError('Requested subjects are not in the prepared bundle')
    if folds and any(f < 1 or f > settings['n_folds'] for f in folds):
        raise ValueError('Requested folds are not in the prepared bundle')
    selected = [j for j in jobs if j['protocol'] in protocols and
                (not subjects or j['subject_id'] in subjects) and
                (not folds or int(j['job_id'].rsplit('_', 1)[-1]) in folds)]
    if not selected:
        raise ValueError('No generation jobs selected')
    # Preflight architecture/source compatibility before training any job.
    # Finalization examines both protocols, including previously completed jobs.
    for job in jobs:
        run = root / job['run_directory']
        if (run / 'config.json').exists():
            saved = load_config(run)
            if saved.num_global_tokens != config.num_global_tokens:
                raise ValueError('Saved global-token count differs from this generation; use a fresh bundle')
    for position, job in enumerate(selected, 1):
        run = root / job['run_directory']
        local = replace(config, subjects=(job['subject_id'],))
        exists = (run / 'config.json').exists()
        if exists and load_config(run) != local:
            raise ValueError(f'Generation settings changed for {job["job_id"]}; use the original settings or a new bundle')
        if verify_generation(root, job):
            print(f'Verified completed job: {job["job_id"]}', flush=True)
            continue
        print(f'Generation job {position}/{len(selected)}: {job["job_id"]}', flush=True)
        run_training(root / job['input_directory'], run, local, resume=resume and exists, generate=True)
        if not verify_generation(root, job):
            raise ValueError(f'Job did not complete: {job["job_id"]}')
    status = finalize(root)
    print(json.dumps(status, indent=2), flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol', choices=(*PROTOCOLS, 'both'), default='both')
    parser.add_argument('--subjects', nargs='+', type=int, help='Generator subjects; not held-out cross-subject fold IDs')
    parser.add_argument('--folds', nargs='+', type=int, help='Within-subject generator fold numbers')
    parser.add_argument('--epochs', type=int, help='Diffusion epochs per model; reference default 1000')
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    parser.add_argument('--resume', action='store_true', help='Resume unfinished jobs with identical settings')
    parser.add_argument('--finalize-only', action='store_true', help='Verify outputs and publish any complete protocol manifests')
    args = parser.parse_args(argv)
    try:
        if args.finalize_only:
            print(json.dumps(finalize(args.bundle), indent=2))
        else:
            generate(args.bundle, protocols=PROTOCOLS if args.protocol == 'both' else (args.protocol,),
                     subjects=args.subjects, folds=args.folds, epochs=args.epochs, device=args.device, resume=args.resume)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
