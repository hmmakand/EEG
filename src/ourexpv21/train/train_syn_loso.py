"""Train Version21 PLV classifiers on finalized LOSO folds."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'src.ourexpv21.train'

from ..dataset import default_dataset_config
from .config_syn import SynTrainingConfig
from .experiment_syn_loso import run_experiment
from .reporting import save_results
from .setup import select_device

BUNDLE = Path(__file__).resolve().parents[1] / 'datasyn/fold_cross'
# LOSO evaluates the final epoch, so the schedule matters: these values were
# chosen on held-out training subjects; longer training overfits them.
# The temporal encoder overfits the synthetic trials, so they are left out.
LOSO_DEFAULTS = dict(num_epochs=20, batch_size=32, ensemble=3, center_logits=True, use_synthetic=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE, help='Prepared signal bundle with finalized cross_subject.csv')
    parser.add_argument('--output', type=Path, help='Results JSON only')
    parser.add_argument('--subjects', type=int, nargs='+', help='Subset of saved subjects; default uses all')
    parser.add_argument('--epochs', dest='num_epochs', type=int, help='Default: 20')
    parser.add_argument('--batch-size', type=int, help='Default: 32')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--model', choices=('custom', 'gat'), help='Classifier (default: custom)')
    parser.add_argument('--paths', dest='model_paths', nargs='+', choices=('graph', 'wide', 'temporal'),
                        help='Logit paths of the custom model to keep (default: all three)')
    parser.add_argument('--ensemble', type=int, help='Models per fold, trained with consecutive seeds (default: 3)')
    parser.add_argument('--synthetic', dest='use_synthetic', action='store_true', default=None,
                        help='Also train on the synthetic trials of the saved folds (default: originals only)')
    parser.add_argument('--no-center-logits', dest='center_logits', action='store_false', default=None,
                        help="Do not remove the held-out subject's mean logit before the decision")
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)
    overrides = {k: getattr(args, k) for k in ('num_epochs', 'batch_size', 'device', 'model', 'ensemble', 'center_logits', 'use_synthetic') if getattr(args, k) is not None}
    if args.subjects is not None:
        overrides['subjects'] = tuple(args.subjects)
    if args.model_paths is not None:
        overrides['model_paths'] = tuple(args.model_paths)
    config = replace(SynTrainingConfig(), **{**LOSO_DEFAULTS, **overrides}, show_progress=not args.quiet)
    dataset_config = default_dataset_config()
    try:
        config.validate()
        results = run_experiment(args.bundle, select_device(config.device), dataset_config, config)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or (Path(__file__).resolve().parents[1] / 'outputs' /
        f'Version21_GritGPS_LOSO_{"Synthetic" if config.use_synthetic else "OriginalOnly"}_{config.num_epochs}epochs_Batch{config.batch_size}_{stamp}.json')
    save_results(results, output)


if __name__ == '__main__':
    main()
