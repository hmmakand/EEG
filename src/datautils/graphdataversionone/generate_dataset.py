"""Generate the local Liu2024 dataset; run with python -m ...generate_dataset."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import hashlib
import importlib.metadata
import json
import os
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
from threadpoolctl import threadpool_limits

from . import config
from .edge_features import build_edge_features, build_edge_index, compute_connectivity
from .metadata import build_dataset_metadata, build_montage_metadata, file_sha256, generation_settings
from .node_features import build_node_features
from .preprocessing import attach_analysis_montage, compute_csd, extract_trial_pair, prepare_eeg
from .source import discover_recordings, load_recording, read_trial_table
from .storage import save_dataset
from .validation import validate_features, validate_saved_dataset, validate_trial_pair


def calculation_fingerprint():
    """Hash the implementation, calculation settings and numerical libraries.

    Returns
    -------
    str
        Digest used to reject stale subject checkpoints; reads Python files.
    """
    payload = {'settings': generation_settings(),
               'code': {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))
                        if not p.name.startswith('test_')},
               'software': {key: importlib.metadata.version(key)
                            for key in ('numpy', 'scipy', 'pandas', 'mne', 'mne-connectivity')}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def generate_subject_features(subject, source_path, source_root, checkpoint_dir, fingerprint, limit_trials=40):
    """Calculate and checkpoint every node/edge variant for one subject.

    Parameters
    ----------
    subject : int
        Liu2024 subject identifier.
    source_path, source_root, checkpoint_dir : path-like
        EDF, local dataset root and writable checkpoint directory.
    fingerprint : str
        Settings/software/code identifier required to reuse a checkpoint.
    limit_trials : int
        Number of chronological trials to retain (40 for full generation).

    Returns
    -------
    str
        Completed subject NPZ path with adjacent sample/provenance JSON.

    Raises
    ------
    RuntimeError
        With subject/trial context if loading, feature extraction or validation
        fails. Both branches are retained together; no trial is silently skipped.

    Notes
    -----
    Original files are unchanged. EEG inputs are volts and CSD inputs V/m².
    Checkpoints contain float32 features and int64 labels in chronological order.
    """
    with threadpool_limits(limits=1):
        source_path = Path(source_path)
        checkpoint = Path(checkpoint_dir) / f'sub-{subject:02d}.npz'
        sidecar = checkpoint.with_suffix('.json')
        source_hash = file_sha256(source_path)
        events_path = Path(source_root) / 'files/38516084'
        if not events_path.is_file():
            events_path = events_path.with_suffix('.tsv')
        electrodes_path = Path(source_root) / 'files/38516078'
        if not electrodes_path.is_file():
            electrodes_path = electrodes_path.with_suffix('.tsv')
        signature = {'fingerprint': fingerprint, 'edf_sha256': source_hash,
                     'events_sha256': file_sha256(events_path),
                     'electrodes_sha256': file_sha256(electrodes_path), 'limit_trials': limit_trials}
        if checkpoint.is_file() and sidecar.is_file():
            old = json.loads(sidecar.read_text())
            if old.get('signature') == signature and old.get('checkpoint_sha256') == file_sha256(checkpoint):
                return str(checkpoint)
        started = time.monotonic()
        trial_index = None
        try:
            raw = load_recording(source_path)
            if raw.info['sfreq'] != config.EXPECTED_SFREQ:
                raise ValueError(f"Expected 500 Hz, got {raw.info['sfreq']}")
            trial_table = read_trial_table(raw, subject, source_path).iloc[:limit_trials]
            eeg = attach_analysis_montage(prepare_eeg(raw))
            montage = build_montage_metadata(eeg, source_root)
            csd, sphere = compute_csd(eeg, **config.CSD_SETTINGS)
            edge_index = build_edge_index()
            collected = {f'node_features_{variant}': [] for variant in config.NODE_VARIANTS}
            collected.update({f'edge_attr_{method}_{variant}': []
                              for variant in config.NODE_VARIANTS for method in config.METHODS})
            for record in trial_table.to_dict(orient='records'):
                trial_index = record['trial_index']
                original, transformed = extract_trial_pair(eeg, csd, record['start_sample'], record['stop_sample'])
                context = f'subject {subject}, trial {trial_index}'
                validate_trial_pair(original, transformed, expected_n_samples=record['stop_sample'] - record['start_sample'], context=context)
                nodes, edges = {}, {}
                for variant, trial in zip(config.NODE_VARIANTS, (original, transformed)):
                    nodes[variant] = build_node_features(trial, eeg.info['sfreq'])
                    connectivity = compute_connectivity(trial, eeg.info['sfreq'])
                    features = build_edge_features(connectivity, edge_index)
                    for method in config.METHODS:
                        edges[f'{method}_{variant}'] = features[method]
                validate_features(nodes, edges, edge_index, context=context)
                for variant, values in nodes.items():
                    collected[f'node_features_{variant}'].append(values)
                for variant, values in edges.items():
                    collected[f'edge_attr_{variant}'].append(values)
            arrays = {key: np.asarray(values, dtype=np.float32) for key, values in collected.items()}
            arrays['labels'] = trial_table['label'].to_numpy(dtype=np.int64)
            temp = checkpoint.with_suffix('.tmp.npz')
            # No key here is named "allow_pickle"; the ** unpacking only supplies
            # savez's dynamic array-name keywords, not its allow_pickle parameter.
            np.savez(temp, **arrays)  # type: ignore[arg-type]
            os.replace(temp, checkpoint)
            payload = {
                'signature': signature, 'checkpoint_sha256': file_sha256(checkpoint),
                'samples': trial_table.to_dict(orient='records'), 'montage': montage,
                'provenance': {'subject': subject, 'source_file': str(source_path),
                               'edf_sha256': source_hash, 'events_sha256': signature['events_sha256'],
                               'marker_audit': trial_table.attrs.get('marker_audit', {}),
                               'fitted_sphere_m': list(sphere), 'elapsed_seconds': time.monotonic() - started},
            }
            temp_json = sidecar.with_suffix('.tmp.json')
            temp_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
            os.replace(temp_json, sidecar)
            raw.close()
            eeg.close()
            csd.close()
            return str(checkpoint)
        except Exception as exc:
            raise RuntimeError(f'Subject {subject}, trial {trial_index}: {exc}') from exc


def main(argv=None):
    """Run pilot/full generation, validate staging files and publish the dataset.

    Parameters
    ----------
    argv : sequence of str or None
        Optional CLI arguments. Defaults to 50 local subjects, 40 trials each,
        four workers and the configured output directory. Existing outputs are
        never overwritten; interrupted work resumes from matching checkpoints.

    Returns
    -------
    None
        Writes a dataset and prints progress plus its final validation summary.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=config.SOURCE_ROOT)
    parser.add_argument('--output', type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--limit-trials', type=int, default=40)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.limit_trials <= 40 or args.workers < 1:
        parser.error('limit-trials must be 1..40 and workers must be positive')
    args.source_root = args.source_root.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f'Output already exists; select a new version/path: {args.output}')
    if args.output == config.OUTPUT_ROOT.resolve() and (args.limit_trials != 40 or args.subjects):
        parser.error('Use a separate --output for a subset/pilot dataset')
    recordings = discover_recordings(args.source_root)
    subjects = sorted(set(args.subjects)) if args.subjects else sorted(recordings)
    if not subjects or not set(subjects) <= recordings.keys():
        parser.error('Requested subjects are absent from the local dataset')
    if args.subjects is None and subjects != list(range(1, 51)):
        raise ValueError('Full generation requires all 50 Liu2024 subjects')
    working = args.output.parent / f'.{args.output.name}.working'
    checkpoint_dir = working / 'subjects'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = calculation_fingerprint()
    start = time.monotonic()
    print(f'Generating {len(subjects)} subjects × {args.limit_trials} trials; {args.workers} workers', flush=True)
    checkpoints = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(generate_subject_features, subject, recordings[subject], args.source_root,
                               checkpoint_dir, fingerprint, args.limit_trials): subject for subject in subjects}
        for future in as_completed(futures):
            subject = futures[future]
            checkpoints[subject] = future.result()
            print(f'Completed sub-{subject:02d}: {len(checkpoints)}/{len(subjects)} subjects '
                  f'({time.monotonic() - start:.1f}s)', flush=True)
    staging = working / 'dataset'
    if staging.exists():
        shutil.rmtree(staging)
    metadata_builder = partial(_metadata_for_run, subjects=subjects, fingerprint=fingerprint,
                               expected_trials_per_subject=args.limit_trials)
    save_dataset(staging, [checkpoints[subject] for subject in subjects], build_edge_index(), metadata_builder)
    report = validate_saved_dataset(staging)
    report['elapsed_seconds'] = time.monotonic() - start
    (staging / 'validation_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(args.output)
    shutil.rmtree(working)
    print(f'Saved verified dataset: {args.output}', flush=True)
    print(json.dumps(report, indent=2), flush=True)


def _metadata_for_run(samples, subject_records, montage, arrays, *, subjects, fingerprint,
                      expected_trials_per_subject):
    """Adapt assembled sample records to the dataset metadata builder.

    Parameters
    ----------
    samples, subject_records : list of dict
        Ordered trials and source provenance.
    montage, arrays : dict
        Geometry and saved array schema.
    subjects : sequence of int
        Included subject IDs.
    fingerprint : str
        Calculation identity.
    expected_trials_per_subject : int
        Required per-subject trial count.

    Returns
    -------
    dict
        Dataset metadata, without file side effects.
    """
    return build_dataset_metadata(len(samples), subjects, montage, subject_records, arrays, fingerprint,
                                  expected_trials_per_subject)


if __name__ == '__main__':
    main()
