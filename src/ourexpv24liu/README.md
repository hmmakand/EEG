# Version twenty-four: the version 23 classifier on the Liu 2024 stroke dataset

Version twenty-four is a standalone copy of [version 23](../ourexpv23/README.md)
that can read a second dataset: the motor-imagery recordings of acute stroke
patients published by Liu et al. (2024). The model, the graph
construction and the training are those of version 23. What is new is the
reader for the Liu 2024 files, [data/liu2024.py](data/liu2024.py), and the
small changes that let the rest of the pipeline work with either dataset.

| | Liu 2024 (`--dataset liu2024`, the default) | BCI Competition IV 2a (`--dataset bci2a`) |
|---|---|---|
| Subjects | 47 acute stroke patients (50 published; 3 are duplicates) | 9 healthy subjects (8 by default) |
| Task | Left-hand or right-hand grasp imagery | Left-hand or right-hand imagery |
| Trials per subject | 40 (20 per class) | 144 (72 per class) |
| EEG electrodes | 29 | 22 |
| Sample rate used | 250 Hz (recorded at 500 Hz) | 250 Hz |
| Analysed window | The 4 s of imagery | The 2 s after the cue |
| Files | One EDF per subject, one events table | `A0xT.mat` |

With `--dataset bci2a` the graphs are identical to those of version 23.

Contents:

1. [Run it](#1-run-it)
2. [The Liu 2024 files](#2-the-liu-2024-files)
3. [How a recording becomes trials](#3-how-a-recording-becomes-trials)
4. [From trials to graphs](#4-from-trials-to-graphs)
5. [Code map and what changed from version 23](#5-code-map-and-what-changed-from-version-23)
6. [Results](#6-results)
7. [Things to know before reporting](#7-things-to-know-before-reporting)

## 1. Run it

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv24liu/requirements.txt

LIU=/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv24liu/outputs

# Leave one subject out: 47 folds
python -u -m src.ourexpv24liu.train.run_loso \
  --data-dir "$LIU" --output "$OUT/Version24Liu_LOSO_seed12345.json"

# Within subject: 10 folds for each of the 47 subjects
python -u -m src.ourexpv24liu.train.run_within \
  --data-dir "$LIU" --output "$OUT/Version24Liu_Within_seed12345.json"
```

`--data-dir` is the folder MOABB created (`MNE-liu2024-data`); its `files`
subfolder is accepted too.

Options of both commands:

| Option | Meaning |
|---|---|
| `--dataset {liu2024,bci2a}` | Dataset to read. Default `liu2024`. |
| `--subjects 1 2 3` | Run only these held-out subjects (LOSO) or subjects (within). LOSO still trains on all the others. |
| `--window START STOP` | Analysed window in seconds from the imagery onset. Default `0 4` for liu2024, `0 2` for bci2a. |
| `--exclude-subjects 4 13` | Leave these subjects out of the experiment entirely, from training and from testing. |
| `--all-subjects` | Also use the subjects left out by default: the three duplicated recordings of liu2024 (5, 15, 49), or subject 4 of bci2a. |
| `--seed`, `--device`, `--quiet` | As in version 23. |
| `--folds`, `--split-seed`, `--epochs` | Within-subject run only, as in version 23. |

The BCI IV 2a experiment of version 23 is run with:

```bash
python -u -m src.ourexpv24liu.train.run_loso --dataset bci2a \
  --data-dir /workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a \
  --output "$OUT/Version24Liu_bci2a_LOSO_seed12345.json"
```

Tests: `python -m unittest discover -s src/ourexpv24liu/tests -t .`
(23 tests; two of them read the real Liu 2024 recordings and are skipped when
they are not on the machine).

## 2. The Liu 2024 files

Liu H. et al., "An EEG motor imagery dataset for brain computer interface in
acute stroke patients", Scientific Data 11, 131 (2024),
DOI 10.1038/s41597-023-02787-8.

```
MNE-liu2024-data/files/
  edffile/sub-01/eeg/sub-01_task-motor-imagery_eeg.edf   one per subject, sub-01 ... sub-50
  38516084              events table, the same for every subject
  38516078              electrode positions (not used)
  participants.tsv      age, paralysed side, clinical scores (not used)
```

What one EDF holds, checked on all 50 subjects:

- **160,000 samples at 500 Hz**: 40 epochs of 8 s, one per trial, concatenated.
- **33 channels**: 29 EEG electrodes, the reference CPz (all zeros), two EOG
  channels (HEOL, HEOR) and a marker channel with an empty name.
- **Already preprocessed** by the authors: baseline removed and band-passed at
  0.5–40 Hz.
- **The epochs are not contiguous in time.** Where two epochs meet, the signal
  jumps about 14 times more than between ordinary neighbouring samples.
- **Marker channel**: 1 at the first sample of every epoch, 2 where the
  imagery starts, 3 where it ends 4 s later.

The events table has three rows per trial (instruction, imagery, break). Its
`trial_type` column gives the class: 1 = left hand, 2 = right hand. The
classes alternate: left, right, left, right, …

The 29 electrodes, in the order used for the graph nodes:

```
FP1 FP2 Fz F3 F4 F7 F8 FCz FC3 FC4 FT7 FT8 Cz C3 C4 T3 T4
CP3 CP4 TP7 TP8 Pz P3 P4 T5 T6 Oz O1 O2
```

## 3. How a recording becomes trials

[data/liu2024.py](data/liu2024.py), for one subject:

1. **Read the EDF** with MNE: the 29 EEG channels and the marker channel. CPz
   and the EOG channels are left out.
2. **Read the labels** from the events table: the class of the imagery row of
   each trial, 40 labels.
3. **Split into 40 epochs** of 4,000 samples. Each must begin with marker 1.
4. **Find the imagery onset** of each epoch: its first marker 2.
5. **Halve the sample rate** to 250 Hz by keeping every second sample. The
   published signal has no content above 40 Hz (in five subjects checked, less
   than one millionth of its power lies above 60 Hz), so nothing is lost and
   no extra filter is needed.

The result is 40 trials per subject: `signal [2000, 29]`, the onset sample,
and the label.

### Why the onset is taken from the marker

The imagery does not start at the same time in every epoch:

| Imagery onset in the epoch | Trials (of 2,000) |
|---|---:|
| About 2.0 s (the nominal layout) | 1,929 |
| At the first sample of the epoch | 23 |
| Between 2.2 s and 3.7 s | 48 |

Subject 6 alone has 25 of the late trials. A fixed window at 2–6 s would be
misplaced by up to 2 s for these 71 trials. Every trial does have an imagery
marker, and the end marker follows it by 4.00 s in every trial, so the marker
is reliable.

A few epochs contain stray markers after the real ones; the first imagery
marker of the epoch is used.

### Three recordings are duplicates and are left out

Three subjects of the published dataset carry the EEG of another subject:

| Copy | Same EEG as | Largest difference | Correlation |
|---|---|---:|---:|
| sub-05 | sub-04 | 0 µV (identical) | 1.000000 |
| sub-15 | sub-11 | 16 µV | 0.999920 |
| sub-49 | sub-47 | 8 µV | 0.999996 |

For comparison, two unrelated subjects correlate at about 0.00. The members of
a pair have different entries in `participants.tsv` (age, sex, lesion), so it
is not known which patient a duplicated recording belongs to.

With both members of a pair in the data, leave-one-subject-out trains on a
copy of the test subject. That is how the defect was found: the first run on
all 50 subjects scored 98% to 100% on exactly these six subjects and 51% on
the other 44.

The default subject list therefore has 47 subjects: 5, 15 and 49 are left out
and 4, 11 and 47 are kept. `--all-subjects` restores all 50.

## 4. From trials to graphs

Shared by both datasets, and unchanged from version 23 except for the size:

1. **Band-pass** 8–30 Hz, Butterworth order 5, zero phase, on the whole 8 s
   epoch.
2. **Window**: the 4 s that follow the trial's imagery onset, 1,000 samples.
3. **Euclidean alignment** per subject: all trials of a subject are whitened
   with the inverse square root of that subject's mean covariance. No labels
   are used.
4. **Graph**: 29 nodes; six band powers per node (8–32 Hz in 4 Hz steps);
   all 812 ordered electrode pairs as edges, each with the phase-locking value
   and the covariance of the pair; the windowed signal `[29, 1000]` for the
   temporal encoder.

The model is built for the dataset's size (29 electrodes and 1,000 samples).
Its other settings are those of version 23.

Training is that of version 23: Adam, learning rate 0.001, weight decay 0.001,
batch 32, three models per fold with seeds 12345 to 12347 whose probabilities
are averaged, score after the final epoch.

| | Leave one subject out | Within subject |
|---|---|---|
| Folds | 47, one per subject | 10 per subject |
| Training trials per fold | 1,840 | 36 |
| Test trials per fold | 40 | 4 |
| Epochs | 20 | 60 |
| Logit centring | Yes | No |

## 5. Code map and what changed from version 23

```
data/config.py       DataConfig; liu2024_config() and bci2a_config()
data/recordings.py   Trial; chooses the reader for the configured dataset
data/liu2024.py      NEW: the Liu 2024 reader
data/bci2a.py        the BCI IV 2a reader of version 23
data/signal.py       band-pass, window, alignment
data/graph.py        trial -> graph
model/               the model of version 23
train/               leave-one-subject-out and within-subject runs
tests/               23 tests
```

| File | Change from version 23 |
|---|---|
| [data/liu2024.py](data/liu2024.py) | New. |
| [data/config.py](data/config.py) | `dataset` and `num_channels` fields; the list of duplicated Liu 2024 recordings; the window is given in seconds from the imagery onset; one function per dataset returns its rules. |
| [data/recordings.py](data/recordings.py) | `Trial` has an `onset` field; the reader is chosen by dataset; every trial's window is checked to fit. |
| [data/bci2a.py](data/bci2a.py) | The former `recordings.py`; sets the onset to 2 s (the cue). |
| [data/signal.py](data/signal.py) | The window is placed at the trial's onset. |
| [model/config.py](model/config.py) | `ModelConfig.for_data` sets the number of electrodes and samples from the data rules. |
| [train/](train/) | `--dataset` and `--window` options; the dataset is recorded in the result file. |
| requirements | `mne` is added, to read EDF files. |

## 6. Results

Liu 2024, 47 subjects, seed 12345, score after the final epoch. Chance is 50%.

| Experiment | Accuracy | Balanced accuracy | Macro F1 | Correct trials |
|---|---:|---:|---:|---:|
| Leave one subject out | **51.65%** | 51.65% | 51.45% | 971 of 1,880 |
| Within subject, 10 folds | **53.03%** | 53.03% | 52.49% | 997 of 1,880 |

- **Leave one subject out is not distinguishable from chance.** 971 of 1,880
  correct is 1.4 standard errors above 50%.
- **Within subject is slightly above chance** (2.6 standard errors), with a
  lean toward predicting "right" (1,033 of 1,880 predictions).
- **Per-subject scores scatter as chance would.** A subject has 40 trials, so
  chance alone gives a spread of about 8 points; the observed spread is 10
  (LOSO) and 12 (within subject). The highest LOSO score is 75% (subject 26),
  the highest within-subject score 90% (subject 40).
- **Subjects that do well in one experiment do not do well in the other**
  (for example subject 26: 75% and 47.5%; subject 40: 47.5% and 90%), which
  again points to noise.

For reference, the dataset's authors report within-subject accuracies of 55.6%
(CSP + LDA), 57.6% (FBCSP + SVM), 61.2% and 72.2% for their own methods, on
all 50 subjects and with their own preprocessing and splits. Those figures are
not directly comparable with the table above.

On BCI IV 2a the same model scores about 79% in both experiments (version 23).
Nothing was tuned for the Liu 2024 data: window, band, epochs and model size
are those found on BCI IV 2a.

### Accuracy by subject

| Subject | Leave one subject out | Within subject |
|---|---:|---:|
| 1 | 47.5% | 40.0% |
| 2 | 37.5% | 35.0% |
| 3 | 50.0% | 65.0% |
| 4 | 47.5% | 42.5% |
| 6 | 52.5% | 55.0% |
| 7 | 50.0% | 62.5% |
| 8 | 40.0% | 52.5% |
| 9 | 70.0% | 45.0% |
| 10 | 67.5% | 50.0% |
| 11 | 52.5% | 67.5% |
| 12 | 32.5% | 65.0% |
| 13 | 50.0% | 45.0% |
| 14 | 45.0% | 32.5% |
| 16 | 42.5% | 55.0% |
| 17 | 50.0% | 62.5% |
| 18 | 52.5% | 52.5% |
| 19 | 55.0% | 50.0% |
| 20 | 47.5% | 62.5% |
| 21 | 50.0% | 35.0% |
| 22 | 47.5% | 75.0% |
| 23 | 57.5% | 47.5% |
| 24 | 42.5% | 40.0% |
| 25 | 60.0% | 70.0% |
| 26 | 75.0% | 47.5% |
| 27 | 52.5% | 37.5% |
| 28 | 65.0% | 37.5% |
| 29 | 57.5% | 45.0% |
| 30 | 40.0% | 50.0% |
| 31 | 32.5% | 50.0% |
| 32 | 55.0% | 55.0% |
| 33 | 52.5% | 52.5% |
| 34 | 60.0% | 67.5% |
| 35 | 50.0% | 45.0% |
| 36 | 40.0% | 62.5% |
| 37 | 70.0% | 62.5% |
| 38 | 67.5% | 70.0% |
| 39 | 50.0% | 55.0% |
| 40 | 47.5% | 90.0% |
| 41 | 55.0% | 52.5% |
| 42 | 52.5% | 52.5% |
| 43 | 60.0% | 35.0% |
| 44 | 57.5% | 57.5% |
| 45 | 35.0% | 62.5% |
| 46 | 70.0% | 47.5% |
| 47 | 40.0% | 50.0% |
| 48 | 50.0% | 55.0% |
| 50 | 45.0% | 45.0% |
| **Mean** | **51.65%** | **53.03%** |

### Without the subjects that have motion artefacts (36 subjects)

The dataset's authors list 13 subjects with motion artefacts: 4, 5, 13, 14,
18, 24, 28, 33, 42, 43, 47, 48 and 49. Two of them (5 and 49) are already left
out as duplicates. Leaving out the other 11 as well, with
`--exclude-subjects 4 13 14 18 24 28 33 42 43 47 48`, gives 36 subjects.

| Experiment | 47 subjects | 36 subjects | Correct trials (36) |
|---|---:|---:|---:|
| Leave one subject out | 51.65% | **52.36%** | 754 of 1,440 |
| Within subject, 10 folds | 53.03% | **55.49%** | 799 of 1,440 |

- **Leave one subject out stays at chance** (1.8 standard errors above 50%).
  The models were retrained without the 11 subjects.
- **Within subject is above chance, but only by 5.5 points** (4.2 standard
  errors). Subjects are trained separately, so the 36 scores are those of the
  47-subject run; the mean rises because the 11 excluded subjects had
  averaged 45.0%.
- **Single-subject LOSO scores are not stable.** Between the 47-subject and
  the 36-subject run, a subject's score changed by up to 22.5 points
  (9 trials of 40), although only the training set differed.

The list of excluded subjects comes from the dataset's publication and was
fixed before the run; no subject was excluded because of its score.

### The run on all 50 subjects, for the record

Before the duplicated recordings were found, both experiments were run on all
50 subjects. Leave-one-subject-out gave 56.55% overall: 98% to 100% on the six
subjects that have a copy in the training set (4, 5, 11, 15, 47, 49) and 50.7%
on the other 44. Within subject gave 53.10%. These numbers must not be
reported as results.

### Result files

In `databci/bciupperlimbactionexperiment/ourexpv24liu/outputs/`:

| File | Content |
|---|---|
| `Version24Liu_LOSO_seed12345.json` | Leave one subject out, 47 subjects |
| `Version24Liu_Within_seed12345.json` | Within subject, 47 subjects |
| `Version24Liu_{LOSO,Within}_NoArtefacts36_seed12345.json` | Both experiments without the 11 artefact subjects, 36 subjects |
| `Version24Liu_{LOSO,Within}_All50WithDuplicates_seed12345.json` | The invalid runs on all 50 subjects |
| `pilot_LOSO_heldout1to5_seed12345.json` | First pilot, five held-out subjects, trained on all 50 |

## 7. Things to know before reporting

- **The published dataset contains duplicated recordings** (section 3). Any
  result on all 50 subjects with subject-wise splitting is affected. This
  includes results obtained with MOABB's loader, which returns all 50.
- **The classes alternate strictly** (left, right, left, …) in every subject.
  The class of a trial therefore equals its position parity. The model sees
  each trial alone and is not given its position, but anything that differs
  systematically between odd and even epochs would be confounded with the
  class. This is a property of the published dataset.
- **Trial labels are the same table for all subjects.** The EDF files carry no
  class information of their own; the labels come from the shared events
  table, as in MOABB.
- **MOABB places the trials differently.** It takes the first 40 imagery
  markers of the recording in order. That is the same as here for most
  subjects, but it shifts trials when a recording contains stray markers.
  Numbers obtained with MOABB's loader are therefore not on exactly the same
  windows.
- **23 trials start with the imagery.** Their window begins at the first
  sample of the epoch, so the band-pass filter has no earlier signal to settle
  on. They are kept.
- **The training settings were not tuned for this dataset.** Epochs, learning
  rate, window and model size are those found on BCI IV 2a. In particular a
  within-subject fold has only 36 training trials here.
- **Within-subject folds have 4 test trials.** A subject's accuracy moves in
  steps of 2.5 points, and per-subject figures are noisy.
- **Alignment uses unlabelled test trials**, as in version 23: a subject's
  whitening matrix is computed from all its trials. In leave-one-subject-out
  the logits of the held-out subject are also centred.
- **The patients differ clinically** (paralysed side, lesion site, severity).
  `participants.tsv` is not used; no analysis by patient group has been done.
