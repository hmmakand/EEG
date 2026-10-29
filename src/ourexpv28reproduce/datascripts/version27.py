"""The samples, subjects and folds of version 27 (our method), so that a paper's model can be tested on exactly
the samples our method is tested on.

Re-implemented here from version 27's rules (the package imports nothing from
other versions); the windows and labels were checked to be identical to
version 27's on the real recordings.

    liu2024   46 stroke patients (without the copies 5, 15, 49 and subject 6): imagery (1) against rest (0).
              Two samples per trial, placed by the trial's own imagery onset: 1.9 s from 1.0 s after the
              onset (imagery) and 1.9 s from 4.0 s after it (the break). A trial whose break does not fit
              into the 8 s epoch gives no sample.
    bci2a     9 healthy subjects (all of them): left hand (0) against right hand (1).
              One sample per trial: the 2 s after the cue.
    jia2019   15 stroke patients: left hand (0) against right hand (1). One sample per trial: the whole
              published epoch (6.832 s), with a common average reference, as MOABB and PA-TCNet use it.

How the signal is filtered and scaled belongs to each paper's preparation;
a sample only says which stretch of which trial it is, and its label.
"""
from dataclasses import dataclass, replace

import numpy as np
from sklearn.model_selection import StratifiedKFold

from . import bci2a, jia2019, liu2024
from .trial import SAMPLE_RATE, Trial

DATASETS = ("liu2024", "bci2a", "jia2019")
CLASS_NAMES = dict(liu2024=("rest", "imagery"), bci2a=("left hand", "right hand"),
                   jia2019=("left hand", "right hand"))
SUBJECTS = dict(
    liu2024=tuple(s for s in range(1, 51) if s not in (*liu2024.DUPLICATES, 6)),
    bci2a=(1, 2, 3, 4, 5, 6, 7, 8, 9),
    jia2019=tuple(range(1, 16)),
)
LIU2024_IMAGERY_START, LIU2024_REST_START, LIU2024_SECONDS = 1.0, 4.0, 1.9
BCI2A_SECONDS = 2.0
WITHIN_FOLDS, WITHIN_SPLIT_SEED = 10, 42


@dataclass(frozen=True)
class Sample:
    trial: Trial        # the trial the sample is cut from; trial.signal is the whole stretch that was read
    start: int          # first sample of the analysed stretch in trial.signal (250 Hz)
    length: int         # samples of the analysed stretch (250 Hz)
    label: int          # see CLASS_NAMES
    group: int          # the trial; samples of one trial stay in the same within-subject fold


def subject_file(data_dir, dataset: str, subject: int):
    return dict(liu2024=liu2024, bci2a=bci2a, jia2019=jia2019)[dataset].subject_file(data_dir, subject)


def read_samples(data_dir, dataset: str, subject: int) -> list[Sample]:
    """The samples of one subject, in version 27's order."""
    if dataset == "liu2024":
        imagery, rest, length = (int(round(s * SAMPLE_RATE))
                                 for s in (LIU2024_IMAGERY_START, LIU2024_REST_START, LIU2024_SECONDS))
        samples = []
        for trial in liu2024.read_trials(data_dir, subject):
            if trial.onset + rest + length > len(trial.signal):
                continue                                       # the break is cut off by the end of the epoch
            samples.append(Sample(trial, trial.onset + imagery, length, 1, trial.trial_id))
            samples.append(Sample(trial, trial.onset + rest, length, 0, trial.trial_id))
    elif dataset == "bci2a":
        length = int(round(BCI2A_SECONDS * SAMPLE_RATE))
        samples = [Sample(t, t.onset, length, t.label, t.trial_id) for t in bci2a.read_trials(data_dir, subject)]
    elif dataset == "jia2019":
        samples = []
        for trial in jia2019.read_trials(data_dir, subject):
            referenced = trial.signal - trial.signal.mean(axis=1, keepdims=True)
            samples.append(Sample(replace(trial, signal=referenced), 0, len(trial.signal), trial.label,
                                  trial.trial_id))
    else:
        raise ValueError(f"dataset must be one of {DATASETS}")
    if not samples:
        raise ValueError(f"{dataset} subject {subject}: no usable trial")
    return samples


def fold_indices(groups, labels, folds: int = WITHIN_FOLDS, split_seed: int = WITHIN_SPLIT_SEED):
    """Version 27's within-subject folds: [(training sample indices, testing sample indices), ...].

    All samples of a trial go to the same fold; trials with a single sample are
    stratified by its label. The same arguments give version 27's folds.
    """
    if folds < 2:
        raise ValueError("At least two folds are required")
    groups, labels = np.asarray(groups), np.asarray(labels)
    trials = np.unique(groups)
    members = [np.flatnonzero(groups == trial) for trial in trials]
    strata = [int(labels[rows[0]]) if len(rows) == 1 else 0 for rows in members]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
    return [(np.sort(np.concatenate([members[i] for i in train])), np.sort(np.concatenate([members[i] for i in test])))
            for train, test in splitter.split(np.zeros(len(trials)), strata)]
