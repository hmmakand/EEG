"""Screen ways of preparing the data with a fast linear classifier; no model is trained.

    python -m src.ourexpv26mix.analysis.screen --data-dir /path/to/MNE-liu2024-data --output screen.json

For every combination of

    band        4-8, 8-13, 13-30, 8-30, 4-40 and 1-4 Hz
    window      0-4, 0-2, 2-4 and 1-3 s after the imagery onset (liu2024); 0-2, 0-1, 1-2 and 0.5-2 s (bci2a)
    eye signal  left in, or regressed out of the EEG with the recording's EOG channels
    features    power     log power per electrode
                change    log power minus the log power of the stretch before the onset
                tangent   logarithm of the covariance of the aligned trial
                course    the signal itself at 25 Hz (1-4 Hz band only)

a shrinkage LDA is evaluated leave one subject out, with the scores centred per
test subject as in the model. Two controls are computed with it:

    EOG alone   the same features from the EOG channels only. If these predict
                the class, a result in that band and window may be eye movement.
    shuffled    the whole grid with the labels permuted inside every subject.
                The best of many combinations is above chance by luck alone;
                the highest value of a shuffled grid shows by how much.

All features are scaled per subject without labels. Choosing a combination
from this table and then reporting its accuracy from the same table would be
selection on the test subjects; the table is for deciding what to train.
"""
import argparse
import itertools
import json
from pathlib import Path

import mne
import numpy as np

from ..data import DATASETS, dataset_config, read_trials
from ..data.graph import covariance_logarithm
from ..data.liu2024 import EEG_CHANNELS, cut_trials, read_labels, read_recording, subject_file as liu_file
from ..data.signal import bandpass

BANDS = ((4, 8), (8, 13), (13, 30), (8, 30), (4, 40), (1, 4))
WINDOWS = dict(liu2024=((0.0, 4.0), (0.0, 2.0), (2.0, 4.0), (1.0, 3.0)),
               bci2a=((0.0, 2.0), (0.0, 1.0), (1.0, 2.0), (0.5, 2.0)))
BASELINE = (-1.5, -0.25)        # seconds from the onset; trials without that much signal use the subject's mean
SHRINKAGE = 0.1
COURSE_RATE = 25


def load_subject(data_dir, subject, config):
    """(signals [trials, samples, EEG + EOG channels], onsets, labels); the EOG channels come last."""
    if config.dataset == "bci2a":
        trials = read_trials(data_dir, subject, config)          # config has 25 channels: 22 EEG + 3 EOG
    else:
        path = liu_file(data_dir, subject)
        eeg, marker = read_recording(path)
        raw = mne.io.read_raw_edf(path, preload=True, verbose="error")
        eog = raw.get_data(picks=["HEOL", "HEOR"]).T
        trials = cut_trials(np.concatenate([eeg, eog], axis=1), marker, read_labels(data_dir), subject, config)
    return (np.stack([t.signal for t in trials]), np.array([t.onset for t in trials]),
            np.array([t.label for t in trials]))


def remove_eog(signals, num_eeg, rate):
    """Regress the EOG channels out of the EEG with one set of weights per subject (no labels)."""
    broadband = bandpass(signals, (0.5, 40.0), rate, 3, time_axis=1).reshape(-1, signals.shape[2])
    weights = np.linalg.lstsq(broadband[:, num_eeg:], broadband[:, :num_eeg], rcond=None)[0]
    cleaned = signals.copy()
    cleaned[:, :, :num_eeg] -= signals[:, :, num_eeg:] @ weights
    return cleaned


def scale(features):
    """Remove the subject's mean and scale every feature to unit spread."""
    return (features - features.mean(axis=0)) / features.std(axis=0).clip(min=1e-12)


def subject_features(signals, onsets, band, window, rate, num_eeg):
    """{feature name: [trials, features]} for the EEG, and the same names with ' (EOG alone)' for the EOG channels."""
    filtered = bandpass(signals, band, rate, 3 if band[0] < 4 else 5, time_axis=1)
    start, stop = (int(round(second * rate)) for second in window)
    windows = np.stack([trial[onset + start:onset + stop] for trial, onset in zip(filtered, onsets)])
    power = np.log(windows.var(axis=1))
    b0, b1 = (int(round(second * rate)) for second in BASELINE)
    has_baseline = onsets + b0 >= 0
    baseline = np.full_like(power, np.nan)
    for row in np.flatnonzero(has_baseline):
        baseline[row] = np.log(filtered[row, onsets[row] + b0:onsets[row] + b1].var(axis=0))
    baseline[~has_baseline] = np.nanmean(baseline, axis=0) if has_baseline.any() else power.mean(axis=0)
    features = {}
    for name, channels in (("", slice(0, num_eeg)), (" (EOG alone)", slice(num_eeg, None))):
        features["power" + name] = scale(power[:, channels])
        features["change" + name] = scale((power - baseline)[:, channels])
        if band == (1, 4):
            course = windows[:, ::rate // COURSE_RATE, channels]
            course = course - course.mean(axis=1, keepdims=True)
            features["course" + name] = (course / course.std(axis=(0, 1), keepdims=True)).reshape(len(course), -1)
    # Tangent space: align the subject's EEG (mean covariance = identity), then the logarithm of each covariance.
    eeg = windows[:, :, :num_eeg]
    covariances = np.einsum("tsc,tsd->tcd", eeg, eeg) / eeg.shape[1]
    values, vectors = np.linalg.eigh(covariances.mean(axis=0))
    values = np.maximum(values, values.max() * 1e-10)
    whitener = (vectors / np.sqrt(values)) @ vectors.T
    upper = np.triu_indices(num_eeg)
    weight = np.where(upper[0] == upper[1], 1.0, np.sqrt(2.0))
    tangent = np.stack([covariance_logarithm(whitener @ c @ whitener)[upper] * weight for c in covariances])
    features["tangent"] = tangent - tangent.mean(axis=0)
    return features


def leave_one_subject_out(features, labels):
    """Mean accuracy of a shrinkage LDA; `features` and `labels` are lists with one entry per subject.

    The class sums and scatter matrices are kept per subject, so leaving a
    subject out is a subtraction.
    """
    counts = np.array([[np.sum(y == c) for c in (0, 1)] for y in labels], dtype=float)
    sums = np.array([[x[y == c].sum(axis=0) for c in (0, 1)] for x, y in zip(features, labels)])
    scatter = np.array([x.T @ x for x in features])
    total_counts, total_sums, total_scatter = counts.sum(0), sums.sum(0), scatter.sum(0)
    accuracies = []
    for held_out, (x, y) in enumerate(zip(features, labels)):
        n = total_counts - counts[held_out]
        means = (total_sums - sums[held_out]) / n[:, None]
        within = (total_scatter - scatter[held_out] - sum(n[c] * np.outer(means[c], means[c]) for c in (0, 1)))
        within /= n.sum() - 2
        within += SHRINKAGE * np.trace(within) / len(within) * np.eye(len(within))
        score = x @ np.linalg.solve(within, means[1] - means[0])
        accuracies.append(np.mean((score > np.median(score)) == (y == 1)))       # centred per test subject
    return float(np.mean(accuracies))


def run(data_dir, config, num_eeg, shuffles=5, seed=0, quick=False):
    rate = config.sample_rate
    data = {subject: load_subject(data_dir, subject, config) for subject in config.subjects}
    print(f"Loaded {len(data)} subjects", flush=True)
    labels = [data[s][2] for s in config.subjects]
    rng = np.random.default_rng(seed)
    shuffled = [[rng.permutation(y) for y in labels] for _ in range(shuffles)]
    bands = BANDS[3:4] if quick else BANDS
    windows = WINDOWS[config.dataset][:1] if quick else WINDOWS[config.dataset]
    rows, null_maxima = [], np.zeros(shuffles)
    for eye in ("left in", "regressed out"):
        signals = {s: (remove_eog(data[s][0], num_eeg, rate) if eye == "regressed out" else data[s][0])
                   for s in config.subjects}
        for band, window in itertools.product(bands, windows):
            per_subject = [subject_features(signals[s], data[s][1], band, window, rate, num_eeg)
                           for s in config.subjects]
            for name in per_subject[0]:
                if eye == "regressed out" and "EOG alone" in name:
                    continue
                features = [f[name] for f in per_subject]
                accuracy = leave_one_subject_out(features, labels)
                row = dict(band=band, window=window, eye_signal=eye, features=name, accuracy=accuracy)
                if "EOG alone" not in name:
                    null = [leave_one_subject_out(features, permuted) for permuted in shuffled]
                    row["shuffled"] = float(np.mean(null))
                    null_maxima = np.maximum(null_maxima, null)
                rows.append(row)
            print(f"band {band}, window {window}, eye signal {eye}: done", flush=True)
    return dict(dataset=config.dataset, subjects=list(config.subjects), shrinkage=SHRINKAGE, baseline_seconds=BASELINE,
                rows=rows, highest_of_each_shuffled_grid=null_maxima.tolist(),
                note="Leave one subject out, shrinkage LDA, scores centred per test subject. 'shuffled' is the mean "
                     "over grids with permuted labels; 'highest_of_each_shuffled_grid' is what the best combination "
                     "of a grid reaches when nothing can be learned.")


def report(result):
    rows = result["rows"]
    eeg = sorted((r for r in rows if "EOG alone" not in r["features"]), key=lambda r: -r["accuracy"])
    eog = {(tuple(r["band"]), tuple(r["window"]), r["features"].replace(" (EOG alone)", "")): r["accuracy"]
           for r in rows if "EOG alone" in r["features"]}
    null = result["highest_of_each_shuffled_grid"]
    print(f"\nBest combination of a grid with shuffled labels: {np.mean(null) * 100:.1f}% on average "
          f"(highest {np.max(null) * 100:.1f}%). Values below that are not evidence of a signal.\n")
    print(f"{'band':>8} {'window':>10} {'eye signal':>14} {'features':>9} {'accuracy':>9} {'EOG alone':>10}")
    for r in eeg[:25]:
        key = (tuple(r["band"]), tuple(r["window"]), r["features"])
        alone = f"{eog[key] * 100:9.1f}%" if key in eog else "         -"
        print(f"{r['band'][0]:>4}-{r['band'][1]:<3} {r['window'][0]:>4}-{r['window'][1]:<4}s {r['eye_signal']:>14} "
              f"{r['features']:>9} {r['accuracy'] * 100:8.1f}% {alone}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=[d for d in DATASETS if d != "jia2019"], default="liu2024")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shuffles", type=int, default=5, help="Grids with permuted labels (default: 5)")
    parser.add_argument("--quick", action="store_true", help="One band and one window only, to try the script")
    args = parser.parse_args(argv)
    num_eeg = 22 if args.dataset == "bci2a" else len(EEG_CHANNELS)
    config = dataset_config(args.dataset, num_channels=num_eeg + (3 if args.dataset == "bci2a" else 2),
                            task="left_right")
    result = run(args.data_dir, config, num_eeg, args.shuffles, quick=args.quick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    report(result)
    print(f"\nResult saved to {args.output}")


if __name__ == "__main__":
    main()
