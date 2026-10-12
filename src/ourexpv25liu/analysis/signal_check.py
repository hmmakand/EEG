"""Is there a left/right signal in the trials? Checks that do not involve the model.

    python -m src.ourexpv25liu.analysis.signal_check --data-dir /path/to/MNE-liu2024-data --output check.json

Three checks on exactly the trials, band and window that the model sees:

  1. Classical classifiers, within subject (the same stratified folds as the
     model's within-subject run) and leave one subject out:
       CSP + LDA               six spatial filters, log variance
       tangent space + LR      Riemannian tangent space of the trial covariances
     For leave one subject out the trials are aligned per subject first, as
     they are for the model.
  2. Lateralisation of band power. Imagining a hand movement lowers the
     8-30 Hz power over the opposite motor cortex, so log power(C4) -
     log power(C3) should be higher in right-hand than in left-hand trials.
     Reported per subject and tested across subjects.
  3. Shuffled labels: check 1 repeated with the labels permuted inside every
     subject, as the chance reference of the same procedure.

With `--dataset bci2a` the same checks run on BCI Competition IV 2a, where
the signal is known to be present; that run validates the checks themselves.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pyriemann.tangentspace import TangentSpace
from scipy import linalg, stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..data import DATASETS, align_subject, dataset_config, read_trials
from ..data.liu2024 import EEG_CHANNELS as LIU2024_CHANNELS
from ..data.signal import analysed_signal

BCI2A_CHANNELS = ("Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                  "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz")


def channel_names(dataset: str):
    return LIU2024_CHANNELS if dataset == "liu2024" else BCI2A_CHANNELS


def covariances(windows):
    """[trials, samples, channels] -> [trials, channels, channels]."""
    return np.einsum("tsc,tsd->tcd", windows, windows) / windows.shape[1]


class CSP:
    """Common spatial patterns: the filters whose variance differs most between the two classes."""

    def __init__(self, filters: int = 6):
        self.filters = filters

    def fit(self, covs, labels):
        mean = [covs[labels == label].mean(axis=0) for label in (0, 1)]
        _, vectors = linalg.eigh(mean[0], mean[0] + mean[1])    # ascending share of class 0
        half = self.filters // 2
        self.weights = np.concatenate([vectors[:, :half], vectors[:, -half:]], axis=1)
        return self

    def transform(self, covs):
        variance = np.einsum("cf,tcd,df->tf", self.weights, covs, self.weights)
        return np.log(variance / variance.sum(axis=1, keepdims=True))


def predict(method: str, train_covs, train_labels, test_covs):
    if method == "csp_lda":
        csp = CSP().fit(train_covs, train_labels)
        return LinearDiscriminantAnalysis().fit(csp.transform(train_covs), train_labels).predict(csp.transform(test_covs))
    space = TangentSpace().fit(train_covs)
    classifier = LogisticRegression(max_iter=1000).fit(space.transform(train_covs), train_labels)
    return classifier.predict(space.transform(test_covs))


METHODS = ("csp_lda", "tangent_lr")


def within_subject(covs, labels, method, folds, split_seed):
    """Accuracy of one subject over all its trials, each predicted by the fold that did not train on it."""
    predictions = np.empty_like(labels)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    for train, test in splitter.split(np.zeros(len(labels)), labels):
        predictions[test] = predict(method, covs[train], labels[train], covs[test])
    return float((predictions == labels).mean())


def leave_one_subject_out(covs, labels, method):
    """{subject: accuracy}; `covs` and `labels` are dictionaries by subject."""
    result = {}
    for held_out in covs:
        others = [subject for subject in covs if subject != held_out]
        prediction = predict(method, np.concatenate([covs[s] for s in others]),
                             np.concatenate([labels[s] for s in others]), covs[held_out])
        result[held_out] = float((prediction == labels[held_out]).mean())
    return result


def lateralisation(windows, labels, names):
    """log power(C4) - log power(C3) per trial, compared between the classes of one subject."""
    power = np.log(windows.var(axis=1))
    index = power[:, names.index("C4")] - power[:, names.index("C3")]
    return dict(difference=float(index[labels == 1].mean() - index[labels == 0].mean()),   # right minus left
                auc=float(roc_auc_score(labels, index)))


def summary(values):
    values = np.asarray(list(values))
    return dict(mean=float(values.mean()), std=float(values.std()), min=float(values.min()), max=float(values.max()))


def run(data_dir, config, folds=10, split_seed=42, shuffle_seed=0) -> dict:
    names = list(channel_names(config.dataset))
    raw_covs, aligned_covs, labels, shuffled, lateral = {}, {}, {}, {}, {}
    rng = np.random.default_rng(shuffle_seed)
    for subject in config.subjects:
        trials = read_trials(data_dir, subject, config)
        labels[subject] = np.array([trial.label for trial in trials])
        shuffled[subject] = rng.permutation(labels[subject])
        windows = np.stack([analysed_signal(trial, config) for trial in trials])
        raw_covs[subject] = covariances(windows)
        aligned = np.stack([analysed_signal(trial, config) for trial in align_subject(trials, config)])
        aligned_covs[subject] = covariances(aligned)
        lateral[subject] = lateralisation(windows, labels[subject], names)
        print(f"S{subject}: {len(trials)} trials", flush=True)

    result = dict(dataset=config.dataset, subjects=list(config.subjects), band=config.band,
                  align_paralysed_side=config.align_paralysed_side,
                  window_seconds=config.window_seconds, folds=folds, split_seed=split_seed)
    for name, used in (("true_labels", labels), ("shuffled_labels", shuffled)):
        block = {}
        for method in METHODS:
            within = {s: within_subject(raw_covs[s], used[s], method, folds, split_seed) for s in config.subjects}
            loso = leave_one_subject_out(aligned_covs, used, method)
            block[method] = dict(within_subject=dict(summary(within.values()), per_subject=within),
                                 leave_one_subject_out=dict(summary(loso.values()), per_subject=loso))
            print(f"{name}, {method}: within subject {block[method]['within_subject']['mean']:.4f}, "
                  f"leave one subject out {block[method]['leave_one_subject_out']['mean']:.4f}", flush=True)
        result[name] = block

    differences = np.array([lateral[s]["difference"] for s in config.subjects])
    test = stats.ttest_1samp(differences, 0.0)
    wilcoxon = stats.wilcoxon(differences)
    result["lateralisation"] = dict(
        definition="log power(C4) - log power(C3), mean of right-hand trials minus mean of left-hand trials; "
                   "positive is the expected direction",
        mean_difference=float(differences.mean()), subjects_in_expected_direction=int((differences > 0).sum()),
        t_statistic=float(test.statistic), t_p_value=float(test.pvalue), wilcoxon_p_value=float(wilcoxon.pvalue),
        mean_auc=float(np.mean([lateral[s]["auc"] for s in config.subjects])), per_subject=lateral)
    print(f"Lateralisation: mean difference {differences.mean():+.4f}, expected direction in "
          f"{int((differences > 0).sum())} of {len(differences)} subjects, t-test p = {test.pvalue:.4g}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="liu2024")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-subjects", type=int, nargs="+", default=())
    parser.add_argument("--align-paralysed-side", action="store_true",
                        help="liu2024: mirror right-paralysed patients, so that the classes are affected hand "
                             "and unaffected hand for every patient")
    parser.add_argument("--window", type=float, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--band", type=float, nargs=2, metavar=("LOW", "HIGH"))
    args = parser.parse_args(argv)
    options = {name: tuple(value) for name, value in (("window_seconds", args.window), ("band", args.band)) if value}
    if args.align_paralysed_side:
        options["align_paralysed_side"] = True
    config = dataset_config(args.dataset, **options)
    config = dataset_config(args.dataset, **options,
                            subjects=tuple(s for s in config.subjects if s not in args.exclude_subjects))
    result = run(args.data_dir, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Result saved to {args.output}")


if __name__ == "__main__":
    main()
