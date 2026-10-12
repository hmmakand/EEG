"""Prepare fresh original signals, original-only folds, and diffusion job inputs."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'datasetsynthetic'

from .config import DEFAULT_OUTPUT, PreparationConfig
from .original import save_originals
from .protocols import save_protocols
from .storage import sha256, verify_bundle, write_json
if __package__ == 'datasetsynthetic':
    from dataset.data_io import resolve_data_dir
else:
    from ..dataset.data_io import resolve_data_dir


def prepare(data_dir, output=DEFAULT_OUTPUT, config=PreparationConfig()):
    config.validate()
    data_dir, output = Path(resolve_data_dir(data_dir)), Path(output).resolve()
    sources = {f'A{s:02d}T.mat': sha256(data_dir / f'A{s:02d}T.mat') for s in config.subjects}
    settings = json.loads(json.dumps(asdict(config)))
    if output.exists() and any(output.iterdir()):
        metadata = verify_bundle(output)
        if metadata['config'] != settings or metadata['source_sha256'] != sources:
            raise ValueError('Existing preparation uses different settings or source data; choose a new output')
        print(f'Reused verified signal bundle: {output}', flush=True)
        return metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.fold-first-', dir=output.parent))
    try:
        rows = save_originals(data_dir, staging / 'original', config)
        jobs = save_protocols(staging, rows, config)
        metadata = dict(format_version=2, workflow='fold_first_signals', config=settings,
                        source_directory=str(data_dir), source_sha256=sources,
                        status='prepared', synthetic_generation_complete=False,
                        original_trial_count=len(rows), generator_job_count=len(jobs),
                        original_order='subject, session_id, original_trial_index',
                        within_subject_rule='KFold, shuffle=True, seed from config; synthetic data generated per training fold',
                        cross_subject_rule='leave one subject out; per-subject generators excluded with their subject',
                        classifier_validation='none; classifier epoch selection is a later training setting',
                        diffusion_validation='validation_per_class drawn only from permitted generator inputs',
                        signal_axes=['trial', 'sample', 'channel'],
                        artifact_sha256={str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()})
        write_json(staging / 'metadata.json', metadata)
        if output.exists():
            output.rmdir()
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f'Prepared {len(rows)} original trials and {len(jobs)} generation jobs: {output}', flush=True)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--folds', dest='n_folds', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--samples-per-class', type=int)
    parser.add_argument('--validation-per-class', type=int)
    args = parser.parse_args(argv)
    overrides = {k: getattr(args, k) for k in ('n_folds', 'seed', 'samples_per_class', 'validation_per_class') if getattr(args, k) is not None}
    if args.subjects is not None:
        overrides['subjects'] = tuple(args.subjects)
    try:
        prepare(args.data_dir, args.output, replace(PreparationConfig(), **overrides))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
