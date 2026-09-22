"""Run fixed-partition or original-only KFold supervised graph training."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in (None, '', 'train'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dataset import default_dataset_config
    from datacustom.dataset import ORIGINAL
    from train.config_syn import SynTrainingConfig
    from train.experiment_syn import run_experiment
    from train.reporting import save_results
    from train.setup import select_device
else:
    from ..dataset import default_dataset_config
    from ..datacustom.dataset import ORIGINAL
    from .config_syn import SynTrainingConfig
    from .experiment_syn import run_experiment
    from .reporting import save_results
    from .setup import select_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-dir', type=Path, default=ORIGINAL)
    parser.add_argument('--synthetic-dir', type=Path, help='One completed run directory; omit for original-only baseline')
    parser.add_argument('--output', type=Path, help='Results JSON only; no graph dataset is saved')
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--epochs', dest='num_epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--combine-valid-test', action='store_true',
                        help='Select the best epoch on combined original validation+testing; not an independent final test')
    parser.add_argument('--protocol', choices=('fixed', 'kfold'), default='fixed',
                        help='kfold splits all originals and adds existing synthetic data to every train fold')
    parser.add_argument('--folds', dest='n_folds', type=int)
    args = parser.parse_args(argv)
    overrides = {name: getattr(args, name) for name in ('num_epochs', 'batch_size', 'device', 'n_folds') if getattr(args, name) is not None}
    if args.subjects is not None:
        overrides['subjects'] = tuple(args.subjects)
    config = replace(SynTrainingConfig(), **overrides, show_progress=not args.quiet,
                     combine_valid_test=args.combine_valid_test, protocol=args.protocol)
    if config.combine_valid_test:
        print('Combined evaluation: validation + testing select the best epoch; no independent final test.')
    if config.protocol == 'kfold':
        print('Original-only folds; existing synthetic trials reused for training. Best test epoch selection.')
    dataset_config = default_dataset_config()
    try:
        config.validate()
        device = select_device(config.device)
        results = run_experiment(args.original_dir, args.synthetic_dir, device, dataset_config, config)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    condition = 'augmented' if args.synthetic_dir is not None else 'original'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    protocol = 'COMBINED_BEST_TEST' if config.combine_valid_test else 'FIXED'
    if config.protocol == 'kfold':
        protocol = 'ORIGINAL_KFOLD'
    output = args.output or (Path(__file__).resolve().parents[1] / 'output' /
        f'{protocol}_{condition}_PLV_{dataset_config.graphs.threshold}_{config.num_epochs}epochs_{stamp}.json')
    print(f'Mean test accuracy across subjects: {results["summary"]["accuracy"]:.4f}')
    save_results(results, output)


if __name__ == '__main__':
    main()
