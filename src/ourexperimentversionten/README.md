# Version ten: the GATv2 baseline on PLV graphs

Version ten is the baseline for left-hand versus right-hand motor imagery on
BCI Competition IV 2a: a three-layer GATv2 classifier on a graph of electrodes
connected where their phase-locking value (PLV) is above 0.35.

This is the corrected version. The model and the graph rule are the original
ones. The data processing and the training and evaluation were fixed using
what versions 18 to 23 established, and unused code was removed.

| Experiment | Result | Chance |
| --- | ---: | ---: |
| Leave one subject out (LOSO), mean of five seeds | **0.535** | 0.5 |
| Within subject, 10-fold, one seed | **0.549** | 0.5 |

For comparison, version 23 scores 0.790 and 0.800 with the same trials, the
same signal processing and the same training and evaluation; only the model
and what the graph carries differ.

**The 74.4% previously reported for this version was not real.** See
[What was wrong before](#what-was-wrong-before).

Contents:

1. [Run it](#1-run-it)
2. [Code map](#2-code-map)
3. [The pipeline](#3-the-pipeline)
4. [Results](#4-results)
5. [What was wrong before](#what-was-wrong-before)
6. [What was kept, changed and removed](#6-what-was-kept-changed-and-removed)
7. [How to read the numbers](#7-how-to-read-the-numbers)

## 1. Run it

Use Python 3.10 or newer. From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexperimentversionten/requirements.txt

RAW=/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv10/outputs

# Leave one subject out, about 4 minutes
python -u -m src.ourexperimentversionten.train.run_loso \
  --data-dir "$RAW" \
  --output "$OUT/Version10Fixed_LOSO_seed12345.json" 2>&1 | tee "$OUT/Version10Fixed_LOSO_seed12345.log"

# Within subject, about 15 minutes
python -u -m src.ourexperimentversionten.train.run_within \
  --data-dir "$RAW" \
  --output "$OUT/Version10Fixed_Within_seed12345.json" 2>&1 | tee "$OUT/Version10Fixed_Within_seed12345.log"
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | required | Folder with the `A0xT.mat` recordings |
| `--output` | timestamped file in `./outputs` | Result JSON |
| `--seed` | 12345 | Seed of the first model of each fold |
| `--subjects` | all configured | Subjects to run (held-out subjects for LOSO) |
| `--all-subjects` | off | Use all nine subjects; the default leaves out subject 4 |
| `--device` | `auto` | `auto`, `cpu` or `cuda` |
| `--quiet` | off | Hide the progress bars |
| `--folds`, `--split-seed`, `--epochs` | 10, 42, 60 | Within-subject run only |

Tests (CPU only, a few seconds, no real data needed):

```bash
python -m unittest discover -s src/ourexperimentversionten/tests -t .
```

## 2. Code map

About 900 lines, tests included (about 1,900 before).

| File | Content |
| --- | --- |
| [data/config.py](data/config.py) | `DataConfig`: every rule that turns a recording into graphs |
| [data/recordings.py](data/recordings.py) | Read the MATLAB files and cut out the left/right trials |
| [data/signal.py](data/signal.py) | Band-pass filter, analysed window, per-subject Euclidean alignment |
| [data/graph.py](data/graph.py) | Log band powers, PLV, the thresholded graph |
| [model/gat.py](model/gat.py) | The GATv2 classifier, unchanged |
| [train/config.py](train/config.py) | `TrainConfig`: model width, schedule, optimiser, seeds, evaluation |
| [train/engine.py](train/engine.py) | One training epoch, the seed ensemble, evaluation |
| [train/loso.py](train/loso.py) | The leave-one-subject-out loop |
| [train/within.py](train/within.py) | The within-subject cross-validation loop |
| [train/metrics.py](train/metrics.py) | Accuracy, balanced accuracy, macro F1, confusion matrix |
| [train/run_loso.py](train/run_loso.py), [train/run_within.py](train/run_within.py) | Command-line entry points |
| [tests/](tests/) | 14 tests on a small synthetic recording |

`data/recordings.py`, `data/signal.py`, `train/metrics.py` and the training
loops are the same as in version 23, so the two versions differ only in the
model and in the graph.

## 3. The pipeline

### Trials

| Rule | Value |
| --- | --- |
| Subjects | 1, 2, 3, 5, 6, 7, 8, 9 by default; `--all-subjects` adds subject 4 |
| Sessions | Every session that contains trials: the six motor-imagery runs |
| Channels | The first 22 (EEG) |
| Classes | Label 1 becomes 0 (left hand), label 2 becomes 1 (right hand) |
| Trial | 4 seconds from the trial start at 250 Hz; the cue appears after 2 seconds |
| Count | 144 trials per subject, 72 per class |

### Signal

| Step | Rule |
| --- | --- |
| Filter | Fifth-order Butterworth band-pass, zero-phase, 8 to 30 Hz, on the whole trial |
| Window | Seconds 2 to 4 of the filtered trial (500 samples), the part after the cue |
| Align | For the node features only: every trial of a subject is multiplied by the inverse square root of that subject's mean covariance (Euclidean alignment). No labels are used |

### Graph

| Field | Shape | Content |
| --- | --- | --- |
| `x` | `[22, 6]` | Log band power of each electrode in 8-12, 12-16, 16-20, 20-24, 24-28, 28-32 Hz, from the aligned signal |
| `edge_index` | `[2, E]` | Electrode pairs whose PLV in the window is above 0.35, in both directions, no self-loops |
| `y` | scalar | 0 left hand, 1 right hand |

The PLV is computed on the signal of the physical electrodes, not on the
aligned signal. Alignment decorrelates the channels, and on aligned signals
only 2 to 11 of the 462 possible pairs pass the threshold. On the unaligned
signal about 450 of 462 pass, as in the original version ten. The graph carries
no edge weights; the model could not use them.

### Model

`GAT` in [model/gat.py](model/gat.py), unchanged: three GATv2 layers with 22
hidden channels and 3 heads, GraphNorm after each, mean pooling over the
electrodes, dropout 0.5, and a linear layer to two classes.

### Training and evaluation

| Setting | LOSO | Within subject |
| --- | --- | --- |
| Split | Train on seven subjects (1008 trials), test on the eighth (144) | Stratified 10-fold inside each subject, seed 42; about 130 training trials per fold |
| Epochs | 20, fixed | 60, fixed |
| Batches | 32, shuffled | 32, shuffled |
| Optimiser | Adam, learning rate 0.001, weight decay 0.001 | Same |
| Models per fold | 3, seeds `seed` to `seed + 2`, probabilities averaged | Same |
| Logit centring | On: the held-out subject's mean logit is subtracted | Off |
| Evaluation | Once, after the final epoch | Once per fold, after the final epoch |

Nothing is selected on test trials. The schedules are those of version 23;
they were not re-tuned for this model.

## 4. Results

### LOSO, five seeds

| Seed | 12345 | 100 | 200 | 300 | 400 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 0.530 | 0.530 | 0.516 | 0.550 | 0.551 | **0.535** |

Each run scores 1152 held-out trials. The standard deviation over seeds is
0.015. The mean is a little above chance: about 3.5 points.

### Per subject

| Subject | LOSO, mean of five seeds | Within subject | Version 23 LOSO | Version 23 within subject |
| --- | ---: | ---: | ---: | ---: |
| S1 | 0.526 | 0.604 | 0.810 | 0.826 |
| S2 | 0.514 | 0.569 | 0.708 | 0.764 |
| S3 | 0.504 | 0.542 | 0.907 | 0.889 |
| S5 | 0.522 | 0.535 | 0.686 | 0.646 |
| S6 | 0.535 | 0.451 | 0.744 | 0.660 |
| S7 | 0.546 | 0.521 | 0.664 | 0.701 |
| S8 | 0.551 | 0.528 | 0.900 | 0.965 |
| S9 | 0.582 | 0.646 | 0.903 | 0.951 |
| **Mean** | **0.535** | **0.549** | **0.790** | **0.800** |

The within-subject column is one run (seed 12345), 633 of 1152 trials correct.
Result files: `Version10Fixed_LOSO_seed*.json` and
`Version10Fixed_Within_seed12345.json` in
`databci/bciupperlimbactionexperiment/ourexpv10/outputs/`.

### Why the baseline stays near chance

With the data and the training now correct, what limits this version is what
the model can see:

- **The graph is almost the same for every trial.** About 450 of 462 pairs
  pass the threshold whatever the class, so the edges carry almost no
  information, and there are no edge weights.
- **The model does not know which electrode is which.** It averages over the
  electrodes at the end and has no electrode identity, so it cannot learn that
  the left-right difference sits at particular electrodes.
- **It never sees the signal over time**, only six band powers per electrode.

Version 23 addresses exactly these three points (edge values on every pair,
an electrode embedding and an electrode-aware path, and a temporal encoder on
the signals).

## What was wrong before

The earlier version ten reported 74.44% within-subject accuracy. A check with
its own training code on four subjects (PLV, 249 epochs, 10 folds):

| Labels | Best test epoch (as reported then) | Final epoch |
| --- | ---: | ---: |
| True | 0.743 | 0.518 |
| Shuffled at random | 0.735 | 0.510 |

Shuffled labels carry no information and still scored 0.735. The reported
number came from choosing, for each fold, the epoch with the highest accuracy
on that fold's 14 or 15 test trials. Scored at the final epoch the model was
at chance. All six connectivity methods tried then (PLV, ciPLV, PLI, wPLI,
coherence, consensus) landed between 0.713 and 0.744 for the same reason.

The files in [output/](output/) are those earlier results. They are kept as a
record and should not be used.

Problems found in the earlier version, and what this version does:

| Problem | Evidence | Now |
| --- | --- | --- |
| Best test epoch reported | The table above | Final epoch only; nothing selected on test trials |
| Folds not stratified | Share of right-hand trials per test fold from 0.27 to 0.79 | Stratified folds |
| Half of each trial is before the cue | CSP + LDA: 0.52 on seconds 0-2, 0.79 on seconds 2-4 | Seconds 2-4 only |
| Raw band powers, different scale per subject | Up to 15 times between subjects | Per-subject alignment and log |
| Two of eight bands empty | The 8-30 Hz filter leaves nothing above 32 Hz | Six bands to 32 Hz |
| Rejected pairs stored as `(0, 0)` placeholder edges | About 35 per graph | Only real pairs are stored |
| Training batches not shuffled | In LOSO a batch held one subject | Shuffled |
| Default connectivity `rplv` called with wrong arguments | The electrode distance matrix received the number 0.0 | Removed; PLV only |
| Results did not record their settings | A file could not be traced to its configuration | Every result file contains all settings |

## 6. What was kept, changed and removed

| | |
| --- | --- |
| **Kept unchanged** | The GATv2 model ([model/gat.py](model/gat.py)) with 22 hidden channels and 3 heads; the graph rule "PLV above 0.35"; subjects, sessions, channels and classes; the 8-30 Hz filter |
| **Changed** | Analysed window, node features (aligned, log, six bands), fold splitting, epoch selection, schedule, shuffling, weight decay, three models per fold, logit centring in LOSO |
| **Removed** | Connectivity methods other than PLV (ciPLV, PLI, wPLI, coherence, consensus, reliability-gated PLV); the legacy edge construction and trial interleaving; the best-epoch selection; the old trainers and their tests and fixtures; the `networkx` requirement |

The previous code, including files that were not in git, is saved in
`databci/bciupperlimbactionexperiment/ourexpv10/code_backup_20261007/`.

## 7. How to read the numbers

- **Unlabeled test-subject data is used in LOSO.** Alignment and logit
  centring use the held-out subject's trials without their labels. In the
  within-subject run, alignment uses all trials of the subject, including each
  test fold's, without labels.
- **Quote the five-seed mean for LOSO.** Single runs range from 0.516 to 0.551.
- **The within-subject result is a single seed.** With 144 trials per subject,
  one subject's accuracy has a noise of about 0.04.
- **The settings were not tuned for this model.** The schedules and
  regularisation are version 23's. A schedule tuned for GATv2 might move the
  result slightly; the three limits listed under
  [Why the baseline stays near chance](#why-the-baseline-stays-near-chance)
  would remain.
- **The comparison with version 23 is like for like** in trials, signal
  processing, splits, seeds and evaluation.
