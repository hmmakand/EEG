"""Train Version14 PLV classifiers on finalized LOSO folds."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'src.ourexperimentversionforteen.train'

from ..dataset import default_dataset_config
from .config_syn import SynTrainingConfig
from .experiment_syn_loso import run_experiment
from .reporting import save_results
from .setup import select_device

BUNDLE = Path(__file__).resolve().parents[1] / 'datasyn/fold_cross'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE, help='Prepared signal bundle with finalized cross_subject.csv')
    parser.add_argument('--output', type=Path, help='Results JSON only')
    parser.add_argument('--subjects', type=int, nargs='+', help='Subset of saved subjects; default uses all')
    parser.add_argument('--epochs', dest='num_epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    overrides = {k: getattr(args, k) for k in ('num_epochs', 'batch_size', 'device') if getattr(args, k) is not None}
    if args.subjects is not None:
        overrides['subjects'] = tuple(args.subjects)
    config = replace(SynTrainingConfig(), **overrides, show_progress=not args.quiet)
    dataset_config = default_dataset_config()
    try:
        config.validate()
        results = run_experiment(args.bundle, select_device(config.device), dataset_config, config)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or (Path(__file__).resolve().parents[1] / 'outputs' /
        f'Version14_PLV_LOSO_Synthetic_{config.num_epochs}epochs_Batch{config.batch_size}_{stamp}.json')
    save_results(results, output)


if __name__ == '__main__':
    main()
