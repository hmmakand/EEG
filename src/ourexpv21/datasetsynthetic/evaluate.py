"""Check generated EEG against the originals it was generated from.

For every completed generation job the generator's permitted original trials
are compared with its synthetic trials:

    class signal   CSP + LDA on the classifier's window (8-30 Hz, post-cue),
                   each group Euclidean-aligned on its own trials:
                   real -> real (cross-validated), real -> synthetic,
                   synthetic -> real, synthetic -> synthetic (cross-validated),
                   and the two cross-validated scores before the cue, where
                   real trials carry no class signal.
    realism        cross-validated accuracy of telling real from synthetic
                   trials from five band powers of the unprocessed signal,
                   reported next to always guessing the larger group.
    spectrum       synthetic / real power per band, and the ratio of their
                   trial-to-trial spread of log band power.

Useful synthetic trials score close to real -> real on real -> synthetic and
synthetic -> real, close to chance before the cue, close to the majority guess
on realism, and close to 1 on the spectrum ratios. No model is trained or
loaded; only saved signals are read.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'datasetsynthetic'

from .config import DEFAULT_OUTPUT
from .protocols import PROTOCOLS
from .storage import write_json

if __package__ == 'datasetsynthetic':
    from dataset import default_dataset_config
    from dataset.preprocessing import apply_bandpass
else:
    from ..dataset import default_dataset_config
    from ..dataset.preprocessing import apply_bandpass

BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
CLASS_METRICS = ('real_to_real', 'real_to_synthetic', 'synthetic_to_real', 'synthetic_to_synthetic',
                 'pre_cue_real_to_real', 'pre_cue_synthetic_to_synthetic')


def covariances(signals):
    """[trials, samples, channels] -> [trials, channels, channels]."""
    return np.einsum('ntc,ntd->ncd', signals, signals) / signals.shape[1]


def align(signals):
    """Euclidean alignment of one group of trials on its own mean covariance."""
    values, vectors = np.linalg.eigh(covariances(signals).mean(axis=0))
    return signals @ ((vectors / np.sqrt(values)) @ vectors.T)


class CspLda:
    """Common spatial patterns (three filters per class) and LDA on log variance."""

    def __init__(self, signals, labels, pairs=3):
        covariance = covariances(signals)
        first, second = covariance[labels == 0].mean(axis=0), covariance[labels == 1].mean(axis=0)
        values, vectors = eigh(first, first + second)
        chosen = np.r_[np.arange(pairs), np.arange(len(values) - pairs, len(values))]
        self.filters = vectors[:, chosen]
        self.classifier = LinearDiscriminantAnalysis().fit(self.features(signals), labels)

    def features(self, signals):
        variance = (signals @ self.filters).var(axis=1)
        return np.log(variance / variance.sum(axis=1, keepdims=True))

    def score(self, signals, labels):
        return float(self.classifier.score(self.features(signals), labels))


def cross_validated(signals, labels, folds=10, seed=0):
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return float(np.mean([CspLda(signals[train], labels[train]).score(signals[test], labels[test])
                          for train, test in splitter.split(signals, labels)]))


def log_band_power(signals, sample_rate):
    """[trials, samples, channels] -> [trials, channels, bands]."""
    frequencies, power = welch(signals, fs=sample_rate, nperseg=min(signals.shape[1], int(2 * sample_rate)), axis=1)
    return np.stack([np.log(power[:, (frequencies >= low) & (frequencies < high)].sum(axis=1))
                     for low, high in BANDS], axis=-1)


def compare(real, real_labels, synthetic, synthetic_labels, sample_rate, edges, window, order=5):
    """All measures for one generator. Signals are [trials, samples, channels]."""
    start, stop = (int(round(second * sample_rate)) for second in window)
    filtered_real = apply_bandpass(real, edges, sample_rate, order, time_axis=1)
    filtered_synthetic = apply_bandpass(synthetic, edges, sample_rate, order, time_axis=1)
    cue_real, cue_synthetic = align(filtered_real[:, start:stop]), align(filtered_synthetic[:, start:stop])
    result = dict(
        real_to_real=cross_validated(cue_real, real_labels),
        real_to_synthetic=CspLda(cue_real, real_labels).score(cue_synthetic, synthetic_labels),
        synthetic_to_real=CspLda(cue_synthetic, synthetic_labels).score(cue_real, real_labels),
        synthetic_to_synthetic=cross_validated(cue_synthetic, synthetic_labels))
    if start > real.shape[2]:  # enough pre-cue samples for a full-rank covariance
        result.update(
            pre_cue_real_to_real=cross_validated(align(filtered_real[:, :start]), real_labels),
            pre_cue_synthetic_to_synthetic=cross_validated(align(filtered_synthetic[:, :start]), synthetic_labels))
    power_real, power_synthetic = log_band_power(real, sample_rate), log_band_power(synthetic, sample_rate)
    features = np.concatenate([power_real, power_synthetic]).mean(axis=1)
    source = np.r_[np.zeros(len(real)), np.ones(len(synthetic))].astype(int)
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    result.update(
        real_versus_synthetic=float(cross_val_score(
            classifier, features, source, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()),
        majority_guess=float(max(len(real), len(synthetic)) / len(source)),
        band_power_ratio=[float(value) for value in
                          np.exp(power_synthetic.mean(axis=(0, 1)) - power_real.mean(axis=(0, 1)))],
        band_power_spread_ratio=[float(value) for value in
                                 power_synthetic.mean(axis=1).std(axis=0) / power_real.mean(axis=1).std(axis=0)])
    return result


def summarize(results):
    """Mean of every measure over the evaluated generators."""
    summary = {}
    for name in results[0]:
        values = [result[name] for result in results if name in result]
        summary[name] = [float(value) for value in np.mean(values, axis=0)] if isinstance(values[0], list) \
            else float(np.mean(values))
    return summary


def evaluate(root=DEFAULT_OUTPUT, *, protocol='cross_subject', subjects=None, synthetic_root=None,
             dataset_config=None):
    """Compare every completed job of `protocol` in the bundle at `root`.

    `synthetic_root`, when given, replaces each job's generated file by
    `<synthetic_root>/<job id>/subject_XX.npz` (arrays `signals`, `labels`), so
    trial runs of a generator can be checked without changing the bundle.
    """
    root = Path(root).resolve()
    config = dataset_config or default_dataset_config()
    window = config.preprocessing.window_seconds or (0.0, config.trials.duration_seconds)
    jobs = [job for job in json.loads((root / 'jobs.json').read_text()) if job['protocol'] == protocol
            and (not subjects or job['subject_id'] in subjects)]
    if not jobs:
        raise ValueError('No generation jobs selected')
    results = {}
    for job in jobs:
        subject = job['subject_id']
        name = f'subject_{subject:02d}.npz'
        generated = (Path(synthetic_root) / job['job_id'] if synthetic_root else root / job['run_directory']) / name
        if not generated.exists():
            continue
        with np.load(root / 'original' / name, allow_pickle=False) as arrays:
            permitted = np.isin(arrays['trial_ids'], job['original_trial_ids'])
            real, real_labels = arrays['signals'][permitted].astype(float), arrays['labels'][permitted]
        with np.load(generated, allow_pickle=False) as arrays:
            synthetic, synthetic_labels = arrays['signals'].astype(float), arrays['labels']
        if synthetic.shape[1:] != real.shape[1:] or not np.isfinite(synthetic).all():
            raise ValueError(f'Unexpected synthetic signals for {job["job_id"]}')
        results[job['job_id']] = compare(real, real_labels, synthetic, synthetic_labels, config.sample_rate,
                                         config.preprocessing.edges, window, config.preprocessing.order)
    if not results:
        raise ValueError('No generated signals found for the selected jobs')
    return dict(bundle=str(root), protocol=protocol, synthetic_root=str(synthetic_root) if synthetic_root else None,
                window_seconds=list(window), filter_edges=list(config.preprocessing.edges),
                bands_hz=[list(band) for band in BANDS], jobs=results, summary=summarize(list(results.values())))


def print_report(report):
    summary = report['summary']
    print(f'Bundle: {report["bundle"]}  protocol: {report["protocol"]}  generators: {len(report["jobs"])}')
    window = report['window_seconds']
    print(f'\nClass signal, CSP + LDA accuracy, seconds {window[0]:g}-{window[1]:g} (chance 0.5)')
    header = ('real->real', 'real->syn', 'syn->real', 'syn->syn')
    print(f'{"job":28s} ' + ' '.join(f'{name:>10s}' for name in header) + f' {"real vs syn":>12s}')
    for job, result in list(report['jobs'].items()) + [('mean', summary)]:
        print(f'{job:28s} ' + ' '.join(f'{result[name]:10.3f}' for name in CLASS_METRICS[:4])
              + f' {result["real_versus_synthetic"]:12.3f}')
    if 'pre_cue_real_to_real' in summary:
        print(f'\nBefore the cue (should be chance): real->real {summary["pre_cue_real_to_real"]:.3f}, '
              f'synthetic->synthetic {summary["pre_cue_synthetic_to_synthetic"]:.3f}')
    print(f'Real versus synthetic: {summary["real_versus_synthetic"]:.3f} '
          f'(always guessing the larger group: {summary["majority_guess"]:.3f})')
    print('\nBand (Hz)      synthetic / real power   synthetic / real trial-to-trial spread')
    for (low, high), power, spread in zip(report['bands_hz'], summary['band_power_ratio'],
                                          summary['band_power_spread_ratio']):
        print(f'{low:>3}-{high:<3}        {power:22.2f}   {spread:38.2f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--protocol', choices=PROTOCOLS, default='cross_subject')
    parser.add_argument('--subjects', nargs='+', type=int, help='Generator subjects; default uses all')
    parser.add_argument('--synthetic-root', type=Path,
                        help='Read <root>/<job id>/subject_XX.npz instead of the bundle\'s generated files')
    parser.add_argument('--output', type=Path, help='Also save the full report as JSON')
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.bundle, protocol=args.protocol, subjects=args.subjects,
                          synthetic_root=args.synthetic_root)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print_report(report)
    if args.output is not None:
        write_json(args.output, report)
        print(f'\nReport saved to {args.output}')


if __name__ == '__main__':
    main()
