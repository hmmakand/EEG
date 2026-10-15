# Version twenty-six (mix)

## The default experiment: imagery against rest in stroke patients, 71.2%

Running version 26 without options gives the experiment with the best result
for stroke rehabilitation so far.

```bash
cd /workspaces/EEG && source .venv/bin/activate
LIU=/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv26mix/outputs

python -u -m src.ourexpv26mix.train.run_loso --quiet --data-dir "$LIU" \
  --output "$OUT/Version26Mix_ImageryRest_liu2024_LOSO_seed12345.json"
```

| Liu 2024, imagery against rest, leave one subject out, final epoch, seed 12345 | |
|---|---:|
| Accuracy | **71.24%** |
| Macro F1 | 71.01% |
| Correct samples | 2,606 of 3,660 |
| Chance | 50% |
| A linear classifier on the same samples (tangent space) | 69.3% |

About 2.5 hours on a GTX 1060. Result file:
`Version26Mix_ImageryRest_liu2024_LOSO_seed12345.json` in `$OUT`.

### How the 71% is obtained

1. **Data.** The Liu 2024 stroke dataset: 46 patients, 29 EEG electrodes.
   Left out are the three duplicated recordings (5, 15, 49) and subject 6,
   whose imagery starts so late that the break is cut off in all but one trial.
2. **Two samples from every trial**, placed by the trial's own imagery onset:
   - 1.9 s from the middle of the imagery (1.0 to 2.9 s after the onset),
     labelled "imagery";
   - 1.9 s of the break that follows (4.0 to 5.9 s after the onset), labelled
     "rest".
   Which hand was imagined is ignored. 1,830 trials give 3,660 samples.
3. **Signal preparation.** Band-pass 8-30 Hz, then alignment of every
   patient's signals to a common scale (Euclidean alignment; no labels).
4. **Each sample becomes a graph.** The 29 electrodes are the nodes, with six
   band powers each; every pair of electrodes is an edge with its phase-locking
   value and covariance; the filtered signal itself is kept for the temporal
   encoder.
5. **Model.** The temporal encoder, the graph head with three hybrid blocks,
   and the wide path; their outputs are added.
6. **Training, leave one subject out.** Train on 45 patients, test on the one
   left out, for each of the 46.
   - 20 epochs, fixed in advance; Adam, learning rate 0.001, batch 32.
   - Three models per patient with seeds 12345 to 12347; their probabilities
     are averaged.
   - Target calibration: in the second half of the training the model also
     learns from the test patient's samples, with its own confident and
     physiologically checked predictions as targets. The true labels of the
     test patient are never used.
7. **Scoring.** The prediction after the last epoch. No epoch is chosen with
   test labels; the highest-test-epoch score (76.46%) is recorded for
   comparison only.

### What the result means

The model tells, for a patient it has never seen, whether the patient is in
the imagery period or in the break. During the imagery the patient watches a
video of a grasping hand, and during the break a still picture. A linear
classifier separates the two with the motor electrodes alone at 65.7%, with
the occipital and parietal electrodes alone at 65.2%, and with the two EOG
channels alone at 61.1%. The 71% therefore reflects imagery together with
visual processing and eye activity, not motor imagery alone. Details are in
section 12 below.

### The other experiments of this version

| Experiment | Command option | Result of the method (final epoch) |
|---|---|---:|
| Liu 2024, imagery against rest (the default) | none | 71.24% |
| Liu 2024, left against right | `--task left-right` | 54.31% (seed 12345), at or near chance |
| BCI IV 2a, left against right | `--dataset bci2a` | 78.47% with the current default of three hybrid blocks (seed 12345); 79.90% with `--layers 2` (five seeds) |
| Jia 2019, left against right | `--dataset jia2019` | 47.75%; within-subject results are not valid there |

## Where this version comes from

Version twenty-six is a standalone copy of [version 24](../ourexpv24liu/README.md).
It imports nothing from other versions. On Liu 2024 and BCI IV 2a it builds
the same graphs as version 24, and with `--no-target-calibration` it trains as
version 24; by default it now also calibrates on the unlabelled test subject
(section 8 below). What has been added since the copy
is listed in [Changes in version 26](#changes-in-version-26).

It does not contain the options added in version 25 (multi-scale temporal
filters, sparse mask, alignment by paralysed side, signal check); none of
them improved a result.

Version 24, and therefore this version, is the classifier of
[version 23](../ourexpv23/README.md) made able to read a second dataset: the motor-imagery recordings of acute stroke
patients published by Liu et al. (2024). The model, the graph
construction and the training are those of version 23. What is new is the
reader for the Liu 2024 files, [data/liu2024.py](data/liu2024.py), and the
small changes that let the rest of the pipeline work with either dataset.

| Left-against-right experiments | Liu 2024 (`--dataset liu2024 --task left-right`) | BCI Competition IV 2a (`--dataset bci2a`) |
|---|---|---|
| Subjects | 47 acute stroke patients (50 published; 3 are duplicates) | 9 healthy subjects (8 by default) |
| Task | Left-hand or right-hand grasp imagery | Left-hand or right-hand imagery |
| Trials per subject | 40 (20 per class) | 144 (72 per class) |
| EEG electrodes | 29 | 22 |
| Sample rate used | 250 Hz (recorded at 500 Hz) | 250 Hz |
| Analysed window | The 4 s of imagery | The 2 s after the cue |
| Files | One EDF per subject, one events table | `A0xT.mat` |

With `--dataset bci2a` the graphs are identical to those of version 23.

## Changes in version 26

### 1. A third dataset: Jia 2019 (`--dataset jia2019`)

Jia T., "EEG data of motor imagery for stroke", figshare (2019),
DOI 10.6084/m9.figshare.7636301. It is the dataset of `moabb.datasets.Jia2019`.
That loader is part of MOABB 1.8, which is not released on pip (the
environment has 1.5.0), so the files are read directly by
[data/jia2019.py](data/jia2019.py); MOABB is not needed.

| | Jia 2019 |
|---|---|
| Subjects | 15 stroke patients |
| Trials per subject | 80 (40 left hand, 40 right hand) |
| Electrodes | 63, without names |
| Sample rate | 512 Hz, resampled to 250 Hz |
| Files | `exp1-S<n>-left.mat` and `exp1-S<n>-right.mat`; each has two blocks (`DATA1`, `DATA2`) of 20 epochs |
| Epoch | 3,500 samples (subjects 1-8) or 4,000 (subjects 9-15); the first 3,500 are used |
| Location | `databci/eegdata/jia2019/` (1.4 GB; `fetch.sh` there downloads the files) |

```bash
JIA=/workspaces/datadisk/databci/eegdata/jia2019
python -u -m src.ourexpv26mix.train.run_loso   --dataset jia2019 --data-dir "$JIA" --output "$OUT/Version26Mix_jia2019_LOSO_seed12345.json"
python -u -m src.ourexpv26mix.train.run_within --dataset jia2019 --data-dir "$JIA" --output "$OUT/Version26Mix_jia2019_Within_seed12345.json"
```

What the publication does not provide, and what was done about it:

- **Channel names.** The order of the 63 electrodes is unknown. The model
  does not need names; analyses that need a particular electrode (C3, C4) are
  not possible.
- **The time of the cue.** Estimated from the recordings without the labels:
  averaged over all trials, the subjects show an evoked response peaking 3.1
  to 3.3 s into the epoch, followed by a drop of the 8-30 Hz power. The onset
  is set to 3.0 s and the 3.5 s after it are analysed (875 samples).
- **The recording order.** The two classes are in separate files. Whether
  they were recorded interleaved or in separate runs is not stated; see the
  warning below.

One change to shared code: the alignment now tolerates a recording that lacks
a dimension (subject 6 has rank 62 of 63); such directions are set to zero
instead of being amplified. For full-rank recordings nothing changes.

### 2. Results on Jia 2019, and why the within-subject result is not valid

Seed 12345, final epoch, 15 subjects, 1,200 trials.

| Experiment | This model | Tangent space + logistic regression |
|---|---:|---:|
| Within subject, 10 folds | 94.08% | 96.3% |
| Leave one subject out | **47.75%** | 47.9% |

**The within-subject figure does not measure motor imagery.** Checks with the
linear classifier, mean over the 15 subjects:

| What is classified | Accuracy |
|---|---:|
| Left vs right, within subject | 96.3% |
| Block 1 vs block 2 inside one class (no class difference exists) | 78.7% |
| Left vs right, trained on one block, tested on the other | 82.8% |
| Left vs right, leave one subject out | 47.9% |

- 96% with a linear classifier is far beyond what motor imagery gives (78% on
  BCI IV 2a with healthy subjects).
- Two blocks of the same class can be told apart at 79%. Recordings therefore
  differ strongly for reasons unrelated to the class, and the class is tied to
  the file it is stored in.
- The "class" pattern is not stable inside a subject: trained on one block and
  tested on the other, subject 1 scores 43% and subject 6 scores 5%.
- Across subjects nothing transfers: leave one subject out is at chance for
  both methods, with single subjects far below 50% (subject 10: 12.5% with the
  model, 6% with the linear classifier).

The consistent reading is that left-hand and right-hand trials were recorded
or processed separately, and that a classifier trained within a subject
recognises the recording, not the imagined hand. **Only leave-one-subject-out
results are meaningful on this dataset, and there the model is at chance.**

For comparison, Wang et al. (2026, arXiv:2604.16554) report leave-one-subject-
out accuracies on this dataset of 55.8% (EEGNet), 57.3% (ShallowConvNet) and
72.8% (their method, which adapts to the test subject); the selection of the
number of epochs is not described there.

No duplicated recordings were found: no trial occurs twice and the largest
correlation between two subjects is 0.04.

Per subject (model):

| Subject | Leave one subject out | Within subject |
|---|---:|---:|
| 1 | 36.3% | 93.8% |
| 2 | 57.5% | 97.5% |
| 3 | 47.5% | 100.0% |
| 4 | 53.8% | 100.0% |
| 5 | 38.8% | 100.0% |
| 6 | 56.3% | 90.0% |
| 7 | 42.5% | 71.3% |
| 8 | 60.0% | 86.3% |
| 9 | 61.3% | 92.5% |
| 10 | 12.5% | 100.0% |
| 11 | 40.0% | 90.0% |
| 12 | 55.0% | 100.0% |
| 13 | 36.3% | 92.5% |
| 14 | 68.8% | 100.0% |
| 15 | 50.0% | 97.5% |
| **Mean** | **47.75%** | **94.08%** |

Result files, in `databci/bciupperlimbactionexperiment/ourexpv26mix/outputs/`:
`Version26Mix_jia2019_{LOSO,Within}_seed12345.json` and `Jia2019_checks.json`.

### 3. Highest-test-epoch score, subject lists and the shuffled-label control

Added for a like-for-like comparison with work that reports differently.

- **Every run now also records the score of the best test epoch.** The result
  is still the score after the final epoch (`accuracy`, unchanged from
  version 24). Next to it, each fold has `highest_epoch` (the metrics of the
  epoch with the highest test accuracy, chosen with the test labels) and
  `test_accuracy_per_epoch`; the summary has `highest_epoch` too.
- `--use-subjects 2 5 8 ...` runs the experiment on exactly these subjects.
- `--shuffle-labels` permutes the training labels: a control in which nothing
  can be learned.
- `--epochs` is available for the leave-one-subject-out command as well.

### 4. Comparison with PA-TCNet on Liu 2024

PA-TCNet (Wang et al. 2026, arXiv:2604.16554; code in `src/temp/PA-TCNet-main`)
reports leave-one-subject-out accuracy of 66.56% on Liu 2024 ("XW-Stroke"),
and 61.56% for EEGNet and 62.29% for ShallowConvNet. Reading its code and
paper shows how those numbers are obtained:

- **The epoch is chosen on the test subject.** In `Model/train.py` the
  held-out subject is the validation set: the checkpoint with the highest test
  accuracy is saved, reloaded and reported, and early stopping watches the
  test loss. Up to 200 epochs are allowed.
- **24 of the 50 subjects are used**, selected by clinical criteria
  (right-handed, first stroke, NIHSS below 10): 2, 5, 8, 9, 11, 12, 14, 17, 21,
  23, 24, 26, 27, 28, 30, 32, 33, 37, 38, 43, 44, 47, 49, 50.
- **That list contains the duplicated pair 47/49** (the same recording, see
  section 3 of this README's dataset part). Subject 5, a copy of subject 4, is
  in the list without subject 4 and is therefore a single recording there.

Our model under the same conditions, leave one subject out, seed 12345:

| Subjects | Labels | Epochs | Final epoch | Highest test epoch |
|---|---|---:|---:|---:|
| Their 24, with the pair 47/49 | real | 20 | 58.44% | 64.58% |
| The 23 distinct subjects (49 removed) | real | 20 | **53.70%** | 61.74% |
| The 23 distinct subjects | shuffled | 20 | 51.41% | 61.09% |
| The 23 distinct subjects | real | 100 | 52.17% | 64.78% |
| The 23 distinct subjects | shuffled | 100 | 52.28% | 63.48% |
| Our 47 subjects | real | 20 | **51.81%** | 59.95% |
| Our 47 subjects | shuffled | 20 | 50.69% | 59.26% |

What the table shows:

- **Scored their way on their subjects, this model reaches 64.6%**: above their
  EEGNet and ShallowConvNet figures and 2 points below PA-TCNet.
- **The duplicated pair is worth about 5 points.** Subjects 47 and 49 both
  score 100%; removing 49 lowers the final-epoch mean from 58.4% to 53.7%.
- **Choosing the best test epoch is worth 8 to 13 points, and it is not
  signal.** With shuffled labels the highest-epoch score is 61.1% after 20
  epochs and 63.5% after 100; with real labels it is 61.7% and 64.8%. The
  difference between real and shuffled labels is about 1 point.
- **More epochs raise the highest-epoch score, not the model.** From 20 to 100
  epochs the final-epoch accuracy goes from 53.7% to 52.2%, while the
  highest-epoch score goes from 61.7% to 64.8%.
- **The result without selection is 53.7% on the 23-subject cohort** (2.2
  standard errors above 50%) and 51.8% on our 47 subjects.

For BCI IV 2a the corresponding figures are in
[version 23, reporting the highest test epoch](../ourexpv23reportinghighest/README.md):
80.0% final epoch, 82.2% highest epoch, and 50.0% / 56.6% with shuffled labels.

A mid-60s leave-one-subject-out accuracy on Liu 2024 can thus be produced with
no usable left/right signal, by the duplicated pair and best-epoch selection
together. PA-TCNet's remaining components (the state-space model and the
adaptation on the test subject) have not been tested here.

Result files: `Version26Mix_liu2024_LOSO_*` in the output folder.

### 5. Target calibration (`--target-calibration`; the default since section 8)

The adaptation step of PA-TCNet (PGTC), built into the leave-one-subject-out
training; see [train/calibration.py](train/calibration.py). It changes the
training only. Model, data processing and scoring stay as they are.

1. **Warm-up**, 10 of the 20 epochs: ordinary training on the other subjects.
2. **Calibration**, the remaining 10: every step also trains on a batch of the
   test subject's unlabelled trials, with the model's own prediction as the
   target, at 5% of the loss. A prediction becomes a target only if
   - the model's probability for it is at least 0.90 (after centring), and
   - the trial's motor-area features match the template of the predicted
     class, built from the training subjects.
   The targets are refreshed after every epoch.

Motor-area features ([data/roi.py](data/roi.py)): 8-30 Hz log power of the
mean signal over FC3/C3/CP3 (left), FC4/C4/CP4 (right) and FCz/Cz (midline),
and the left-right difference; computed before alignment and standardised
within the subject. Datasets without channel names (jia2019) cannot use the
option.

No label of the test subject is used. Like the alignment and the logit
centring, the step assumes the test subject's trials are available as a group.

Differences from PA-TCNet's code: the motor-area features are band powers
(theirs are raw signals averaged over electrodes), and the targets are
refreshed once per epoch from all test trials (theirs after every batch). The
three settings are their code defaults and were not tuned.

**Results. Both columns are this method, scored at the final epoch.**

BCI IV 2a, leave one subject out, eight subjects:

| Seed | Calibration off | Calibration on | Change |
|---|---:|---:|---:|
| 12345 | 80.03% | 80.99% | +0.95 |
| 100 | 78.56% | 79.51% | +0.95 |
| 200 | 78.13% | 79.25% | +1.13 |
| 300 | 79.08% | 79.69% | +0.61 |
| 400 | 79.34% | 80.03% | +0.69 |
| **Mean** | **79.03%** | **79.90%** | **+0.87** |

Liu 2024, leave one subject out, 47 subjects, seed 12345:

| Calibration off | Calibration on |
|---:|---:|
| 51.81% | 51.60% |

- **On BCI IV 2a the gain is small but present in all five seeds**
  (+0.6 to +1.1 points; paired t-test over the seeds p < 0.001). About 39% of
  a test subject's trials receive a target, and 88% of the targets are right.
- **On Liu 2024 nothing changes.** 12.6% of the trials receive a target and
  49% of the targets are right. The step reinforces what the model already
  finds; it cannot create a signal.
- By subject on BCI IV 2a (mean of five seeds, off to on): S1 81.0 to 81.5,
  S2 70.8 to 70.7, S3 90.7 to 92.2, S5 68.6 to 69.0, S6 74.4 to 75.8,
  S7 66.4 to 68.3, S8 90.0 to 89.9, S9 90.3 to 91.7.

The "off" figures for BCI IV 2a are the five runs of version 23, whose
training version 26 reproduces without the option. Result files:
`Version26Mix_TargetCalibration_*`.

### 6. Which path earns the accuracy (`--paths`), and work on the graph head (`--node-signal`)

`--paths graph temporal` runs the model with only the named paths (an
ablation). **The rows below are ablations, except the last one, which is the
method itself.** BCI IV 2a, leave one subject out, final epoch, five seeds
(12345, 100, 200, 300, 400):

| Paths | Mean | Seed spread (sd) |
|---|---:|---:|
| Graph head only | 72.73% | 0.49 |
| Temporal encoder only | 79.38% | 1.18 |
| Graph head + temporal encoder | 78.94% | 1.16 |
| All three (the method, calibration off) | 79.03% | 0.73 |

- **The temporal encoder alone equals the full model** (79.38% against
  79.03%; paired over seeds, p = 0.24).
- **The graph head and the wide path add nothing to it.** The single-run
  table of version 21, which suggested about one point, was seed noise.
- The graph head alone is stable at 72.7%, 6.6 points below the temporal
  encoder.

The graph head sees six band powers per electrode, each one number for the
whole window; the temporal encoder sees the signal. `--node-signal` gives the
graph nodes features learned from the electrode's own signal (a temporal
convolution shared by all electrodes, then the log power per filter; no
spatial filter, so combining electrodes is left to the graph layers):

| Graph head only, with | Mean of five seeds | Change |
|---|---:|---:|
| Band powers (as in the method) | 72.73% | |
| `--node-signal power`: plus log power per learned filter, whole window | 73.40% | +0.67 |
| `--node-signal timed`: plus log power per filter and time step | 75.09% | +2.36 |

Time-resolved learned node features raise the graph head by 2.4 points, in
every seed. It remains 4.3 points below the temporal encoder alone.

Result files: `Version26Mix_Paths-*`.

### 7. Two further changes to the graph head, and the result on both datasets

- `--node-signal mixed`: as `timed`, but before the power is taken every node
  mixes the filtered signals of all nodes. The weight of source j in a mixture
  of node i is a learned weight for that pair of electrodes plus a term from
  the trial's edge features (phase locking, covariance). This is spatial
  filtering done on the graph. Two mixtures per node.
- `--readout electrode`: the graph head also keeps four values of every
  electrode in the electrode's place, instead of only the mean over electrodes.

Leave one subject out, final epoch. BCI IV 2a: mean of five seeds (12345, 100,
200, 300, 400). Liu 2024: 47 subjects, seed 12345.

**Ablations (paths switched off, no calibration):**

| Paths and node features | BCI IV 2a | Liu 2024 |
|---|---:|---:|
| Graph head only, band powers (as in version 23) | 72.73% | 50.32% |
| Graph head only, `timed` | 75.09% | |
| Graph head only, `timed` + electrode readout | 76.46% | |
| Graph head only, `mixed` | 77.93% | **54.73%** |
| Graph head only, `mixed` + electrode readout | 77.78% | |
| Graph head only, `mixed`, with target calibration | 77.74% | |
| Temporal encoder only | 79.38% | 51.17% |

**The method (all three paths):**

| Node features | Target calibration | BCI IV 2a | Liu 2024 |
|---|---|---:|---:|
| Band powers | off (version 23) | 79.03% | 51.81% |
| Band powers | **on (the default since this section)** | **79.90%** | 51.60% |
| `timed` | off | 78.89% | |
| `timed` | on | 79.44% | |
| `mixed` | off | 78.26% | 52.34% |
| `mixed` | on | 78.72% | |

What this shows:

- **The signal-level graph step makes the graph head a working model on its
  own**: 72.7% to 77.9% on BCI IV 2a, above CSP + LDA (76.0%) and 1.5 points
  below the temporal encoder.
- **Inside the full model it does not help.** With all three paths, `mixed`
  is about one point below band-power nodes on BCI IV 2a (p = 0.01 without
  calibration, p = 0.04 with it): the graph head and the temporal encoder then
  learn the same thing from the same signal.
- **On Liu 2024 the graph head with `mixed` gives the highest figure
  measured there, 54.7%** (1,029 of 1,880 trials, 4.1 standard errors above
  50%), against 50.3% to 52.3% for every other configuration. Over three
  seeds it is 53.8% (table below).
- The electrode readout adds a point to `timed` and nothing to `mixed`.
- Target calibration helps the full model and not the graph head alone.

Three seeds on Liu 2024 for the two configurations of interest (47 subjects,
leave one subject out, final epoch):

| Configuration | Seed 12345 | Seed 100 | Seed 200 | Mean |
|---|---:|---:|---:|---:|
| The method, default (all paths, calibration on) | 51.60% | 53.35% | 50.85% | 51.93% |
| Ablation: graph head only, `mixed` | 54.73% | 54.10% | 52.55% | **53.79%** |

The graph head with `mixed` is ahead in every seed, by 1.9 points on average.
Compared per subject over the 47 subjects the difference is not significant
(p = 0.11; the graph head is better for 29 of 47). Both are close to chance:
53.8% is the highest three-seed figure measured on this dataset, and it is
still a weak result.

Number of hybrid blocks in the graph head (`--layers`; graph head only with
`mixed`, BCI IV 2a, leave one subject out, **seed 12345 only**):

| Blocks | Parameters | Accuracy |
|---:|---:|---:|
| 1 | 43,450 | 77.00% |
| 2 (the setting) | 60,922 | 75.87% |
| 3 | 78,394 | 77.95% |
| 4 | 95,866 | 77.69% |

The four figures lie within the spread between seeds of this configuration
(two blocks: 75.87% to 79.51% over five seeds, mean 77.93%), so no depth is
shown to be better. Depth is not what limits the graph head; two blocks are
kept.

### 8. The default configuration of the model and the training

**Since this section the default of leave-one-subject-out runs is: all three
paths, band-power nodes, target calibration on** (and, since section 9, three
hybrid blocks instead of two). It has the best five-seed
mean on BCI IV 2a (79.90%) and is not worse on Liu 2024 than the previous
default (51.60% against 51.81%, well inside one standard error).

- `--no-target-calibration` gives the training of versions 23 and 24 exactly.
- The warm-up is half of the epochs (10 of 20).
- Target calibration is switched off automatically for jia2019 (no channel
  names) and is not used in within-subject runs.

The configuration was chosen among those in the tables above on the test
results themselves, which favours it slightly; the gain of the calibration
over five seeds (+0.87 points) is the more careful figure than the best single
number.

The graph head alone with `mixed` is the alternative to consider when the
model is to be a graph model: 77.9% on BCI IV 2a (five seeds) and 53.8% on
Liu 2024 (three seeds), with

```bash
python -u -m src.ourexpv26mix.train.run_loso --paths graph --node-signal mixed --no-target-calibration ...
```

### 9. Three hybrid blocks as the default, and a temporal transformer on the nodes

**The number of hybrid blocks in the graph head is 3 by default since this
section** (`--layers 2` gives the two blocks of versions 23 to 25 and of all
results in the sections above). The value was requested after the depth table
of section 7; measured afterwards on the full model, seed 12345 only, leave
one subject out, final epoch:

| The method (all three paths, target calibration) | BCI IV 2a | Liu 2024 |
|---|---:|---:|
| 2 blocks | 80.99% | 51.60% |
| 3 blocks (the default) | 78.47% | 54.31% |

On BCI IV 2a three blocks is below every one of the five seeds run with two
blocks (79.25% to 80.99%). On Liu 2024 it is 2.7 points higher, about two
standard errors on one seed; the two-block figure itself ranged from 50.85% to
53.35% over three seeds. Which depth to keep is an open decision.

`--node-time-attention 1` (the temporal transformer of EEGformer, Wan et al.
2023): the time steps of every node's signal features attend to each other
before the hybrid blocks. Needs `--node-signal timed` or `mixed`. Seed 12345,
BCI IV 2a:

| Configuration | Without | With time attention |
|---|---:|---:|
| Ablation: graph head only, `mixed`, 3 blocks | 77.95% | 77.86% |
| The method, 3 blocks, `mixed`, target calibration | | 79.77% |

No gain for the graph head. Of EEGformer's three transformers, the one over
electrodes is what the pair attention of the hybrid block already does; the one
over the filters of an electrode was not built.

### 10. Two new kinds of features (optional; measured, neither is adopted)

- `--tangent-features`: the matrix logarithm of the trial's covariance. After
  alignment the subject's mean covariance is the identity, so this is the
  covariance in the tangent space at the subject's mean. Its diagonal becomes
  a seventh node feature and its off-diagonal entries a third edge feature;
  through the wide path the whole tangent vector reaches a linear classifier.
  A linear classifier on tangent-space features alone gave 54.2% on Liu 2024
  and 73.4% on BCI IV 2a (version 25).
- `--low-frequency`: a fourth path that classifies from the 0.5-4 Hz
  potentials, which the 8-30 Hz band-pass removes. The trial is band-passed to
  0.5-4 Hz (Butterworth order 3, zero phase), aligned per subject in that band,
  reduced to 25 Hz, and each channel's mean over the window is removed. Eight
  learned spatial filters, then a linear layer on the time courses; the path
  starts at zero.

Commands, seed 12345 (about 12 minutes each on BCI IV 2a, 2.5 hours on Liu 2024):

```bash
cd /workspaces/EEG && source .venv/bin/activate
BCI=/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a
LIU=/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv26mix/outputs
RUN="python -u -m src.ourexpv26mix.train.run_loso --quiet"

$RUN --dataset bci2a --data-dir "$BCI" --tangent-features --output "$OUT/Version26Mix_Tangent_bci2a_LOSO_seed12345.json"
$RUN --dataset bci2a --data-dir "$BCI" --low-frequency    --output "$OUT/Version26Mix_LowFrequency_bci2a_LOSO_seed12345.json"
$RUN --dataset bci2a --data-dir "$BCI" --tangent-features --low-frequency --output "$OUT/Version26Mix_Tangent+LowFrequency_bci2a_LOSO_seed12345.json"
$RUN --data-dir "$LIU" --tangent-features --output "$OUT/Version26Mix_Tangent_liu2024_LOSO_seed12345.json"
$RUN --data-dir "$LIU" --low-frequency    --output "$OUT/Version26Mix_LowFrequency_liu2024_LOSO_seed12345.json"
$RUN --data-dir "$LIU" --tangent-features --low-frequency --output "$OUT/Version26Mix_Tangent+LowFrequency_liu2024_LOSO_seed12345.json"
```

Results, the method with its default settings plus the option, seed 12345,
leave one subject out, final epoch:

| Option | BCI IV 2a | Liu 2024 |
|---|---:|---:|
| None (the default) | 78.47% | 54.31% |
| `--tangent-features` | 79.60% | 51.33% |
| `--low-frequency` | **86.28%** | 53.09% |
| Both | 86.20% | 54.31% |

**The 86% on BCI IV 2a is not motor imagery. It is mostly eye movement.**
The cue of BCI IV 2a is an arrow pointing left or right, and the recording has
three EOG channels that this pipeline does not use. A linear classifier
(shrinkage LDA, leave one subject out, per-subject scaling) on the 0.5-4 Hz
time courses of the 2 s after the cue gives:

| Channels | Accuracy | Subjects 5, 6, 7 |
|---|---:|---|
| The 3 EOG channels alone | 77.8% | 97%, 78%, 93% |
| All 22 EEG electrodes | 73.6% | 90%, 76%, 82% |
| Central electrodes (C5 ... C6) | 57.1% | |
| C3 and C4 | 56.2% | |
| 22 EEG electrodes after regressing the EOG channels out | 71.5% | 76%, 75%, 72% |

- The EOG channels alone predict the class better than the EEG does, and the
  two outer EOG channels deflect in opposite directions in the first half
  second after the cue: the eyes follow the arrow.
- The subjects in which the low-frequency path gained most (5, 6 and 7: +23,
  +13 and +22 points) are the ones in which the EOG channels alone reach 97%,
  78% and 93%.
- Over the motor cortex the low band carries little (56% to 57%).
- After a linear regression of the EOG channels the EEG still gives 71.5%.
  Three monopolar EOG channels do not remove eye movements completely, and
  the response of the visual cortex to a left or right arrow is a second
  class-dependent signal that is not imagery either. What remains has not
  been shown to be motor-related.

The same classifier on parts of the 2 s after the cue shows that the slow-band
information exists only while the eyes move:

| Window after the cue | EOG channels alone | 22 EEG electrodes | EEG, EOG regressed out |
|---|---:|---:|---:|
| 0.0-0.5 s | 79.7% | 75.7% | 73.6% |
| 0.5-1.0 s | 67.2% | 57.5% | 61.5% |
| 1.0-1.5 s | 61.8% | 53.8% | 53.3% |
| 1.5-2.0 s | 55.7% | 50.0% | 52.8% |

The EEG follows the EOG channels in time and falls to chance once the eyes are
still, although the imagery continues. A motor-imagery signal would not end
half a second after the cue.

**The 8-30 Hz band is not affected in this way.** The same classifier on the
8-30 Hz log power gives 68.4% for the 22 EEG electrodes as recorded and 72.2%
after regressing the EOG channels out; the EOG channels alone give 66.7%. The
results of this method in 8-30 Hz do not rest on eye movements.

On Liu 2024 neither option helps (the cue there is a video of a grasping
hand, not an arrow). `--tangent-features` gains one point on BCI IV 2a, within
seed noise, and loses three on Liu 2024.

Neither option is adopted. `--low-frequency` must not be used to report a
motor-imagery result on BCI IV 2a.

### 11. Screening of data preparations for Liu 2024 ([analysis/screen.py](analysis/screen.py))

A shrinkage LDA, leave one subject out, on 6 bands x 4 windows x eye signal
left in or regressed out x up to 4 kinds of features, with two controls: the
same features from the EOG channels alone, and the whole grid with shuffled
labels. No model is trained; about 30 minutes.

```bash
python -W ignore -m src.ourexpv26mix.analysis.screen --data-dir "$LIU" --output "$OUT/Screen_liu2024.json"
python -W ignore -m src.ourexpv26mix.analysis.screen --dataset bci2a --data-dir "$BCI" --output "$OUT/Screen_bci2a.json"
```

| | Liu 2024 | BCI IV 2a |
|---|---:|---:|
| Best combination | 55.9% (4-40 Hz, 0-4 s, tangent, EOG regressed out) | 73.4% (8-30 Hz, 0.5-2 s, tangent, EOG regressed out) |
| The setting of this version (8-30 Hz, full window, tangent) | 55.1% | 72.2% |
| Best combination of a grid with shuffled labels | 53.5% on average, 54.3% at most | 54.2% on average |

- **On Liu 2024 no preparation stands out.** The best of about 130
  combinations is 55.9%, two points above what the best combination reaches
  with shuffled labels, and less than one point above the current setting.
  Band, window, baseline normalisation and removal of the eye signal do not
  uncover a stronger left/right signal.
- The EOG channels alone are at chance on Liu 2024 (44% to 53%), so the
  left/right results there are not eye movement.
- On BCI IV 2a the current band and window are confirmed, removing the eye
  signal adds about one point, and the 1-4 Hz band shows the eye-movement
  effect of section 10 (EOG alone 75.5%).

Left against right in fixed parts of the 8 s trial of Liu 2024 (tangent
features, 8-30 Hz): instruction 0-2 s 56.8%, imagery 2-6 s 54.6%, break
6-8 s 54.6%, **the whole trial 0-8 s 51.6%**. Using the whole trial, as Zhang
et al. (2026) do for stroke severity, does not help here: severity is a
property of the patient and is present in every part of the recording, the
imagined hand is not.

### 12. A second task for Liu 2024: imagery against rest (`--task imagery-rest`)

Two samples per trial, placed by the trial's own imagery onset: 1.9 s of the
imagery starting 1 s after the onset (label 1) and 1.9 s of the break
starting 4 s after the onset (label 0). The imagined hand is not used. A
trial whose break does not fit into the epoch gives no sample: 1,831 of 1,880
trials remain. Subject 6, whose imagery starts late in almost every trial,
keeps 1 trial and is left out.

**This task is the default experiment of the version** since this section:
`--dataset liu2024` without `--task` runs it, with subject 6 left out. The run
below was made with `--task imagery-rest --exclude-subjects 6`, which is the
same configuration; `--task left-right` gives the experiment of the sections
above.

```bash
python -u -m src.ourexpv26mix.train.run_loso --quiet --data-dir "$LIU" \
  --output "$OUT/Version26Mix_ImageryRest_liu2024_LOSO_seed12345.json"
```

**Result of the method** (default settings: three blocks, all paths, target
calibration; seed 12345; leave one subject out, 46 subjects, final epoch):

| | Accuracy | Macro F1 | Correct samples |
|---|---:|---:|---:|
| Imagery against rest | **71.24%** | 71.01% | 2,606 of 3,660 |

41 of the 46 subjects are at 60% or above and 29 at 70% or above; the range is
49% to 89%. Rest is taken for imagery more often (644 samples) than imagery
for rest (410). Target calibration gives a pseudo-label to 45% of a test
subject's samples, 80% of them right. For comparison only, the
highest-test-epoch score is 76.46%.

What a linear classifier finds (tangent features, 8-30 Hz, leave one subject
out, shuffled labels 50%):

| Electrodes | Accuracy |
|---|---:|
| All 29 (the reference for the result above) | 69.3% |
| Motor (FC3 FC4 C3 C4 Cz CP3 CP4 FCz) | 65.7% |
| Occipital and parietal (O1 Oz O2 P3 P4 Pz T5 T6) | 65.2% |
| Frontal (FP1 FP2 F7 F8 F3 F4 Fz) | 62.7% |
| The two EOG channels alone (log power) | 61.1% |

The signal is clear, unlike left against right. It is not specific to the
motor cortex: during the imagery the patient watches a video and during the
break a still picture, and the occipital electrodes and the EOG channels
separate the two almost as well as the motor electrodes. The task therefore
measures "engaged in the video-guided imagery" against "break", which is a
mixture of imagery, visual processing and eye movement.

### 13. What works and what does not: the record so far

Leave one subject out, final epoch, unless stated. "BCI" is BCI IV 2a (eight
subjects), "Liu" is Liu 2024 (47 subjects).

**Works**

| What | Evidence |
|---|---|
| The analysed window after the cue, per-subject Euclidean alignment, the temporal encoder, logit centring, the seed ensemble | Versions 18 to 23: from 50% to 79% on BCI |
| Final-epoch scoring with a fixed number of epochs | With shuffled labels it gives chance; best-test-epoch scoring gives 57% to 69% with shuffled labels |
| Target calibration on the unlabelled test subject | +0.87 points on BCI over five seeds, in every seed; nothing on Liu |
| The temporal encoder as the source of the accuracy | Alone 79.4% on BCI (five seeds), equal to the full model |
| Time-resolved learned node features (`timed`) | Graph head alone: 72.7% to 75.1% on BCI |
| The signal-level graph step (`mixed`) | Graph head alone: 72.7% to 77.9% on BCI (five seeds), above CSP + LDA; 53.8% on Liu (three seeds), the highest figure there |
| Leaving out the duplicated recordings of Liu 2024 | With them, six subjects score 98% to 100% by leakage |
| Imagery against rest on Liu 2024 | 71.2% with the method, 69.3% with a linear classifier, 50% with shuffled labels; not specific to the motor cortex |

**Does not work (no gain, or worse)**

| What | Where measured |
|---|---|
| The graph head and the wide path on top of the temporal encoder | BCI, five seeds: 79.0% with all three, 79.4% with the temporal encoder alone |
| The improved graph head (`timed`, `mixed`) inside the full model | BCI: `mixed` is about 1 point worse than band-power nodes |
| Electrode-aware readout on top of `mixed` | BCI: 77.8% against 77.9% |
| More hybrid blocks in the graph head alone | BCI, one seed: 1, 2, 3, 4 blocks within seed noise |
| Three hybrid blocks in the full model | BCI: 78.5% against 81.0% on the same seed (Liu: 54.3% against 51.6%) |
| Temporal transformer on the node features (EEGformer) | BCI: 77.9% with and without |
| Multi-scale temporal filters, learned sparse electrode mask (ASA-STGCN) | Version 25, both datasets |
| Tangent-space covariance as node and edge features | BCI 79.6% against 78.5% (noise); Liu 51.3% against 54.3% |
| A path on the 0.5-4 Hz potentials | Liu: no gain. BCI: +7.8 points, but from eye movements following the arrow cue, so not valid |
| Target calibration for the graph head alone | BCI: 77.7% against 77.9% |
| Alignment by paralysed side | Version 25, Liu |
| Leaving out the artefact subjects | Version 24, Liu: LOSO stays at chance |
| Other bands, windows, baseline-normalised power, removal of the eye signal, the whole 8 s trial | Liu, linear classifier: best 55.9%, two points above the best of a shuffled grid |
| Cropped training, temporal node and edge modes in the full model, adversarial, contrastive and Proxy-Anchor losses, test-time adaptation without the motor-area check, synthetic trials from diffusion models | Versions 20 to 22, BCI |
| PLV-threshold graphs with GAT layers | Version 10: 53% to 56% |

**About the data**

| Finding | Detail |
|---|---|
| Liu 2024 holds very little left/right signal in 8-30 Hz | CSP + LDA 50% to 52%; the expected C3/C4 power shift in 25 of 47 patients; every model between 50% and 55% |
| Liu 2024 has three duplicated recordings | sub-05 = sub-04, sub-15 = sub-11, sub-49 = sub-47; in the raw files too, with white noise added |
| Jia 2019 within-subject results are not valid | The classes are in separate files and a classifier recognises the file: 94% to 96% within subject, 48% across subjects |
| Best-test-epoch scoring and the duplicated pair explain mid-60s figures on Liu 2024 | This model reaches 64.6% under PA-TCNet's conditions and 53.7% without them |
| On BCI IV 2a the 0.5-4 Hz band after the cue is dominated by eye movements | The three unused EOG channels alone give 77.8% there; in 8-30 Hz the EEG is not affected (72.2% after removing EOG, 68.4% before) |
| Single-seed differences below about 1.5 points on BCI are not reliable | The spread between seeds is 0.7 to 1.3 points |

### The three datasets side by side (this model, final epoch)

| Dataset | Subjects | Leave one subject out | Within subject | Note |
|---|---:|---:|---:|---|
| BCI IV 2a (healthy) | 8 | 79.9% (default; 79.0% without target calibration) | about 80% | Mean of five seeds |
| Liu 2024 (stroke) | 47 | 51.7% | 53.0% | Three duplicated recordings left out |
| Jia 2019 (stroke) | 15 | 47.8% | 94.1% (not valid) | Class tied to the recording file |

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
python -m pip install -r src/ourexpv26mix/requirements.txt

LIU=/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv26mix/outputs

# The default: imagery against rest, leave one subject out, 46 folds
python -u -m src.ourexpv26mix.train.run_loso \
  --data-dir "$LIU" --output "$OUT/Version26Mix_ImageryRest_liu2024_LOSO_seed12345.json"

# The same task within subject: 10 folds for each of the 46 subjects
python -u -m src.ourexpv26mix.train.run_within \
  --data-dir "$LIU" --output "$OUT/Version26Mix_ImageryRest_liu2024_Within_seed12345.json"

# Left against right, as in version 24: 47 subjects
python -u -m src.ourexpv26mix.train.run_loso --task left-right \
  --data-dir "$LIU" --output "$OUT/Version26Mix_LeftRight_liu2024_LOSO_seed12345.json"
```

`--data-dir` is the folder MOABB created (`MNE-liu2024-data`); its `files`
subfolder is accepted too.

Options of both commands:

| Option | Meaning |
|---|---|
| `--dataset {liu2024,jia2019,bci2a}` | Dataset to read. Default `liu2024`. |
| `--subjects 1 2 3` | Run only these held-out subjects (LOSO) or subjects (within). LOSO still trains on all the others. |
| `--window START STOP` | Analysed window in seconds from the imagery onset. Default `0 1.9` for liu2024 imagery-rest, `0 4` for liu2024 left-right, `0 3.5` for jia2019, `0 2` for bci2a. |
| `--paths graph temporal` | Ablation: use only these paths of the model (graph, wide, temporal). |
| `--node-signal {off,power,timed,mixed}` | Give the graph nodes features learned from the electrode signals. Default off. |
| `--readout {mean,electrode}` | How the graph head reads out the nodes. Default mean. |
| `--layers N` | Hybrid blocks in the graph head. Default 3 (versions 23 to 25: 2). |
| `--node-time-attention N` | Transformer blocks over the time steps of every node's signal features. Default 0. |
| `--task {left-right,imagery-rest}` | What is classified. Default `imagery-rest` for liu2024 (the only dataset that has it), `left-right` for bci2a and jia2019. |
| `--tangent-features` | Add tangent-space covariance to the node and edge features. Off by default. |
| `--low-frequency` | Add a path that classifies from the 0.5-4 Hz potentials. Off by default. |
| `--target-calibration` / `--no-target-calibration` | Leave one subject out only: after a warm-up, also train on the unlabelled test subject with checked pseudo-labels. **On by default.** |
| `--use-subjects 2 5 8` | Run on exactly these subjects (training and testing). |
| `--shuffle-labels` | Control run: permute the training labels. |
| `--epochs` | Epochs per model (default 20 for leave one subject out, 60 within subject). |
| `--exclude-subjects 4 13` | Leave these subjects out of the experiment entirely, from training and from testing. |
| `--all-subjects` | Also use the subjects left out by default: for liu2024 the three duplicated recordings (5, 15, 49) and, in imagery-rest, subject 6; for bci2a subject 4. |
| `--seed`, `--device`, `--quiet` | As in version 23. |
| `--folds`, `--split-seed` | Within-subject run only, as in version 23. |

The BCI IV 2a experiment of version 23 is run with:

```bash
python -u -m src.ourexpv26mix.train.run_loso --dataset bci2a \
  --data-dir /workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a \
  --output "$OUT/Version26Mix_bci2a_LOSO_seed12345.json"
```

Tests: `python -m unittest discover -s src/ourexpv26mix/tests -t .`
(44 tests; three of them read real recordings and are skipped when those are
not on the machine).

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
data/jia2019.py      NEW in version 26: the Jia 2019 reader
data/roi.py          NEW in version 26: motor-area features
train/calibration.py NEW in version 26: target calibration
analysis/screen.py   NEW in version 26: screening of data preparations with a linear classifier
tests/               44 tests
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

**These are the results of version 24.** Version 26 has the same code, so the
same commands reproduce them to within the usual run-to-run variation of the
GPU, but no run has been made with version 26 yet. The result files listed
below are in the output folder of version 24.

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
