# Version twenty-seven: the method of our best result, on three datasets

Version twenty-seven contains only the method that produces our best result
for stroke rehabilitation, and runs it on three datasets and with two
evaluation protocols. Everything else that versions 23 to 26 could do has been
left out; it is listed in [What is not in this version](#9-what-is-not-in-this-version).

| Dataset | Task | Subjects | Leave one subject out |
|---|---|---:|---:|
| Liu 2024, acute stroke patients (`--dataset liu2024`, the default) | Imagery against rest | 46 | **71.24%** |
| BCI Competition IV 2a, healthy subjects (`--dataset bci2a`) | Left against right hand | 8 | 78.47% |
| Jia 2019, stroke patients (`--dataset jia2019`) | Left against right hand | 15 | not run yet |

Score after the final epoch, seed 12345, chance 50%. The two figures were
measured with version 26 in the configuration that this version reproduces
exactly (see [Results](#7-results)). The within-subject runs, and Jia 2019 with
this configuration, have not been made yet.

Contents:

1. [Run it](#1-run-it)
2. [How the 71% is obtained](#2-how-the-71-is-obtained)
3. [Code map](#3-code-map)
4. [The data, step by step](#4-the-data-step-by-step)
5. [The model](#5-the-model)
6. [Training and evaluation](#6-training-and-evaluation)
7. [Results](#7-results)
8. [What the stroke result means, and what it does not](#8-what-the-stroke-result-means-and-what-it-does-not)
9. [What is not in this version](#9-what-is-not-in-this-version)
10. [Where this version comes from](#10-where-this-version-comes-from)

## 1. Run it

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv27/requirements.txt

LIU=/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data
BCI=/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv27/outputs

# Liu 2024, imagery against rest
python -u -m src.ourexpv27.train.run_loso   --quiet --data-dir "$LIU" --output-dir "$OUT"   # about 2.5 h
python -u -m src.ourexpv27.train.run_within --quiet --data-dir "$LIU" --output-dir "$OUT"

# BCI IV 2a, left against right hand
python -u -m src.ourexpv27.train.run_loso   --quiet --dataset bci2a --data-dir "$BCI" --output-dir "$OUT"   # about 12 min
python -u -m src.ourexpv27.train.run_within --quiet --dataset bci2a --data-dir "$BCI" --output-dir "$OUT"   # about 25 min

# Jia 2019, left against right hand (the within-subject result is not valid; see 4.3)
JIA=/workspaces/datadisk/databci/eegdata/jia2019
python -u -m src.ourexpv27.train.run_loso   --quiet --dataset jia2019 --data-dir "$JIA" --output-dir "$OUT"
python -u -m src.ourexpv27.train.run_within --quiet --dataset jia2019 --data-dir "$JIA" --output-dir "$OUT"
```

Times are for a GTX 1060.

Figures and the model table, from any result files (see [What else a result records](#what-else-a-result-records-for-the-manuscript)):

```bash
python -m src.ourexpv27.analysis.figures "$OUT"/liu2024/Version27_liu2024_ImageryRest_LOSO_seed12345.json
```

Ablations of the method (leave one subject out; each is one more run of the
same length):

```bash
python -u -m src.ourexpv27.train.run_loso --quiet --data-dir "$LIU" --output-dir "$OUT" --no-target-calibration
python -u -m src.ourexpv27.train.run_loso --quiet --data-dir "$LIU" --output-dir "$OUT" --paths graph
python -u -m src.ourexpv27.train.run_loso --quiet --data-dir "$LIU" --output-dir "$OUT" --paths temporal
python -u -m src.ourexpv27.train.run_loso --quiet --data-dir "$LIU" --output-dir "$OUT" --paths wide
python -u -m src.ourexpv27.train.run_loso --quiet --data-dir "$LIU" --output-dir "$OUT" --paths graph temporal
```

### Where the output goes

One folder per dataset inside the output directory:

```
$OUT/liu2024/Version27_liu2024_ImageryRest_LOSO_seed12345.json
$OUT/liu2024/Version27_liu2024_ImageryRest_LOSO_seed12345.summary.txt   # the same scores as a readable table
$OUT/liu2024/Version27_liu2024_ImageryRest_LOSO_seed12345.embeddings.npz # features of the test samples (t-SNE)
$OUT/liu2024/Version27_liu2024_ImageryRest_LOSO_seed12345_figures/       # made by analysis/figures.py
$OUT/liu2024/Version27_liu2024_ImageryRest_LOSO_NoCalibration_seed12345.json   # an ablation
$OUT/liu2024/Version27_liu2024_ImageryRest_Within_seed12345.json
$OUT/bci2a/Version27_bci2a_LeftRight_LOSO_seed12345.json
$OUT/bci2a/Version27_bci2a_LeftRight_Within_seed12345.json
$OUT/jia2019/Version27_jia2019_LeftRight_LOSO_seed12345.json
$OUT/jia2019/Version27_jia2019_LeftRight_Within_seed12345.json
$OUT/<dataset>/models/<name of the result file>/...        with --save-models
```

If a result file of that name exists, a time stamp is added to the new one;
nothing is overwritten.

### Options

| Option | Meaning |
|---|---|
| `--dataset {liu2024,bci2a,jia2019}` | Default `liu2024`. |
| `--data-dir` | liu2024: the folder MOABB created (`MNE-liu2024-data`, or its `files` subfolder). bci2a: the folder with `A01T.mat` ... `A09T.mat`. jia2019: the folder with `exp1-S1-left.mat` ... (`databci/eegdata/jia2019`; `fetch.sh` there downloads the files). |
| `--output-dir` | Results go to `<output-dir>/<dataset>/`. Default `./outputs`. |
| `--output` | An exact result file, instead of the name chosen in `--output-dir`. |
| `--save-models` | Save every trained model after its final epoch (see below). Off by default. |
| `--subjects 1 2 3` | Run only these subjects. Leave one subject out still trains on all the others. |
| `--seed` | Seed of the first model of each fold. Default 12345. |
| `--device {auto,cpu,cuda}`, `--quiet` | Device; hide the progress bars. |
| `--folds`, `--split-seed`, `--epochs` | Within-subject run only. Defaults 10, 42 and 60. |
| `--paths {graph,wide,temporal} ...` | Ablation: only these paths of the model; their logits are added. Default all three (the method). The result file name gets `_Paths-<paths>`. |
| `--no-target-calibration` | Ablation, leave one subject out only: no target calibration. The name gets `_NoCalibration`. |

On the screen and in the result file: accuracy, balanced accuracy, macro F1, Cohen's kappa, precision, recall and AUC, for the final epoch (the result) and for the highest test epoch (for comparison only); see [Results](#7-results).

### Saved models

`--save-models` writes one file per trained model: `S<subject>_model<k>.pt`
for leave one subject out and `S<subject>_fold<n>_model<k>.pt` within subject
(three models per fold). A file holds the weights after the final epoch and
the settings needed to rebuild the model:

```python
saved = torch.load(path, weights_only=True)
model = EEGGraphModel(ModelConfig(**saved["model_config"]))
model.load_state_dict(saved["state_dict"])
```

- The results do not depend on saved models; nothing is selected or reloaded.
- A model of a leave-one-subject-out fold was trained without one subject and
  then calibrated to that subject. It documents the fold; it is not the model
  to use on a new patient. For that, one model trained on all subjects is
  needed, which this version does not train yet.

Tests: `python -m unittest discover -s src/ourexpv27/tests -t .`
(32 tests; two read real recordings and are skipped when those are not on the
machine).

## 2. How the 71% is obtained

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
   test labels, no checkpoint is selected, there is no validation set.

## 3. Code map

About 2,100 lines without the tests. The package imports nothing from other
versions.

```
datascripts/config.py      every rule that turns a recording into graphs; one function per dataset
datascripts/recording.py   a sample; chooses the reader for the dataset
datascripts/liu2024.py     Liu 2024: read an EDF, split it into epochs, cut the imagery and the rest sample
datascripts/bci2a.py       BCI IV 2a: read a MATLAB file, cut the left-hand and right-hand trials
datascripts/jia2019.py     Jia 2019: read the two class files of a subject, resample to 250 Hz
datascripts/signal.py      band-pass, the analysed stretch, per-subject alignment
datascripts/graph.py       one sample -> one graph
datascripts/roi.py         motor-area features, used by the target calibration
model/config.py            model sizes
model/layers.py            GatedGCN, PairAttention, GraphLayer (the hybrid block), TemporalEncoder
model/model.py             EEGGraphModel: graph head + wide path + temporal encoder
train/config.py            training settings
train/engine.py            one epoch, test logits, the ensemble's decision
train/calibration.py       target calibration on the unlabelled test subject
train/loso.py              leave one subject out; training of one fold; saving of models
train/within.py            within-subject cross-validation
train/metrics.py           accuracy, balanced accuracy, macro F1, confusion matrix
train/output.py            result and model paths, per dataset
train/run_loso.py          command line, leave one subject out
train/run_within.py        command line, within subject
train/complexity.py        parameters, FLOPs, inference time, file size; features for t-SNE
analysis/figures.py        ROC curves, training curves, t-SNE, confusion matrix, model table
tests/                     32 tests
```

## 4. The data, step by step

### 4.1 Liu 2024 ([datascripts/liu2024.py](datascripts/liu2024.py))

Liu H. et al., "An EEG motor imagery dataset for brain computer interface in
acute stroke patients", Scientific Data 11, 131 (2024),
DOI 10.1038/s41597-023-02787-8. One EDF per subject, as downloaded by MOABB:

```
MNE-liu2024-data/files/edffile/sub-01/eeg/sub-01_task-motor-imagery_eeg.edf   ... sub-50
```

- 160,000 samples at 500 Hz: 40 epochs of 8 s, one per trial, concatenated.
  The epochs are not contiguous in time, so each is handled on its own.
- 33 channels: 29 EEG electrodes, the reference CPz (all zeros), two EOG
  channels and a marker channel. The 29 EEG electrodes are used:

  ```
  FP1 FP2 Fz F3 F4 F7 F8 FCz FC3 FC4 FT7 FT8 Cz C3 C4 T3 T4
  CP3 CP4 TP7 TP8 Pz P3 P4 T5 T6 Oz O1 O2
  ```
- Already band-passed at 0.5-40 Hz by the authors.
- The marker channel is 1 at the first sample of every epoch, 2 where the
  imagery starts and 3 where it ends, 4 s later.

A trial is: an instruction, 4 s of imagery during which the patient watches a
video of a grasping hand, and a break with a still picture.

**The two samples of a trial.** The imagery does not start at the same time
in every epoch (2.0 s into the epoch in 1,929 of 2,000 trials, at the epoch
start in 23, between 2.2 and 3.7 s in 48), so every trial's own imagery
marker is used.

| Sample | Stretch, from the trial's imagery onset | Label |
|---|---|---|
| Imagery | 1.0 s to 2.9 s | 1 |
| Rest | 4.0 s to 5.9 s (in the break) | 0 |

- Both stretches are 1.9 s (475 samples at 250 Hz). The length is what
  remains of the break inside the epoch for a trial with the usual timing.
- The first second of the imagery and of the break is left out, where the
  patient reacts to the change on the screen.
- A trial whose break does not fit into the epoch gives no sample, so the
  classes stay balanced. Of the 46 subjects' 1,840 trials, 1,830 remain
  (subject 20 keeps 38, subject 40 keeps 32, all others 40).
- The sample rate is halved to 250 Hz by keeping every second sample; the
  published signal has no content above 40 Hz.

**Subjects that are left out** ([datascripts/config.py](datascripts/config.py)):

| Subject | Reason |
|---|---|
| 5, 15, 49 | Copies of the recordings of subjects 4, 11 and 47 (correlation 1.000). With both members of a pair, leave one subject out trains on the test data. |
| 6 | The imagery starts late in almost every trial; one trial has room for both samples. |

### 4.2 BCI Competition IV 2a ([datascripts/bci2a.py](datascripts/bci2a.py))

Files `A01T.mat` ... `A09T.mat`: 22 EEG electrodes at 250 Hz, four classes of
motor imagery. As in versions 7 to 23:

- Only left-hand (label 0) and right-hand (label 1) trials are used: 144 per
  subject, 72 per class.
- A trial is read from 2 s before the cue to 2 s after it; one sample per
  trial, the 2 s after the cue (500 samples).
- Subjects 1, 2, 3, 5, 6, 7, 8 and 9; subject 4 is left out.

This dataset has no imagery-against-rest task here; it is included as the
dataset on which the method was developed and on which differences between
methods can be measured.

### 4.3 Jia 2019 ([datascripts/jia2019.py](datascripts/jia2019.py))

Jia T., "EEG data of motor imagery for stroke", figshare (2019),
DOI 10.6084/m9.figshare.7636301; `moabb.datasets.Jia2019` in MOABB 1.8.
Fifteen stroke patients imagined moving the left or the right hand.

- Two files per subject, `exp1-S<n>-left.mat` and `exp1-S<n>-right.mat`; the
  class is given by the file. Each file holds two blocks of 20 epochs.
- 63 electrodes of the 10-10 system at 512 Hz, **without channel names**.
  Epochs of 3,500 samples (subjects 1-8) or 4,000 (subjects 9-15); the first
  3,500 are used and resampled to 250 Hz (1,708 samples).
- **Prepared as MOABB and PA-TCNet do, so that results can be compared with
  theirs:** the whole epoch (0 to 6.8 s, 1,708 samples) is analysed, and a
  common average reference is applied (the mean over the 63 electrodes is
  subtracted at every time point).
- The publication does not give the time of the cue. Estimated from the
  recordings without the labels, it is about 3 s into the epoch (an evoked
  response peaks at 3.1 to 3.3 s in every subject); `CUE_SECONDS` in the reader
  keeps this estimate. Until version 27 changed to the whole epoch, the 3.5 s
  after it were analysed.
- 80 samples per subject, 40 per class, one per trial.

Because the electrodes have no names, there are no motor-area features, and
**target calibration is switched off for this dataset** (leave one subject
out prints a note). Everything else is as for the other datasets. Subject 6
has 62 of 63 dimensions; the alignment leaves the missing direction at zero.

**The within-subject result on this dataset is not valid.** The two classes
are in separate files, and the recordings differ for reasons unrelated to the
class: in version 26 a linear classifier told two blocks of the same class
apart at 79%, and left from right within subject at 96%, while left against
right across subjects was 48%. A within-subject classifier recognises the
recording, not the imagined hand. The within-subject command prints a warning;
only leave one subject out is meaningful here.

### 4.4 Signal and graph, all datasets ([datascripts/signal.py](datascripts/signal.py), [datascripts/graph.py](datascripts/graph.py))

1. Band-pass 8-30 Hz, Butterworth order 5, zero phase, on the whole stretch
   that was read; then the analysed stretch is cut out.
2. Euclidean alignment per subject: all samples of a subject are whitened
   with the inverse square root of that subject's mean covariance. No labels.
3. Graph of a sample: one node per electrode with six band powers (8-32 Hz in
   4 Hz steps); every ordered electrode pair as an edge, with the
   phase-locking value and the covariance of the pair; the stretch itself.

| | liu2024 | bci2a | jia2019 |
|---|---:|---:|---:|
| Nodes | 29 | 22 | 63 |
| Edges | 812 | 462 | 3,906 |
| Signal per node | 475 samples | 500 samples | 1,708 samples |
| Common average reference | no | no | yes |

## 5. The model

[model/model.py](model/model.py); 64,862 parameters for liu2024. The same
settings are used for both datasets; only the number of electrodes and the
sample length follow the dataset. The logits are the sum of three paths:

| Path | Input | What it does |
|---|---|---|
| Temporal encoder | The signal | 8 temporal filters (0.1 s), 16 spatial filters, squaring, average over 0.3 s windows, logarithm, dropout 0.7, linear layer |
| Graph head | Band powers, pair features, electrode identity, random-walk encoding | Three hybrid blocks (gated graph convolution + attention over electrode pairs, width 32, 4 heads), mean over nodes and over pairs, a small MLP |
| Wide path | Band powers and pair features, flattened | Dropout 0.5 and one linear layer, starting at zero |

## 6. Training and evaluation

Common to both protocols: Adam, learning rate 0.001, weight decay 0.001,
batch 32; three models per fold (seeds 12345, 12346, 12347) whose class
probabilities are averaged; a fixed number of epochs; the score after the
final epoch.

| | Leave one subject out ([train/loso.py](train/loso.py)) | Within subject ([train/within.py](train/within.py)) |
|---|---|---|
| Split | Train on all other subjects, test on one | 10 folds inside every subject |
| Epochs | 20 | 60 |
| Target calibration | Yes (not for jia2019) | No |
| Logit centring | Yes | No |
| liu2024: samples per fold | 3,580 training, 80 test | 72 training, 8 test |
| bci2a: samples per fold | 1,008 training, 144 test | about 130 training, 14 or 15 test |
| jia2019: samples per fold | 1,120 training, 80 test | 72 training, 8 test |

### 6.1 Target calibration ([train/calibration.py](train/calibration.py))

Leave one subject out only, and only for the datasets with channel names
(liu2024 and bci2a). Epochs 1 to 10 train on the other subjects alone.
After epoch 10 and after every later epoch, the model predicts the test
subject's samples, and a prediction becomes a training target if

- its probability is at least 0.90, and
- the sample's motor-area features match the template of the predicted class.
  The features are the 8-30 Hz log power over FC3/C3/CP3, FC4/C4/CP4 and
  FCz/Cz and the left-right difference; a template is the mean of a class
  over the training subjects.

In epochs 11 to 20 every step also trains on a batch of the test subject's
samples toward these targets, at 5% of the loss. Samples without a target
contribute nothing. The method is the PGTC step of PA-TCNet (Wang et al. 2026),
with band-power features and one refresh per epoch.

### 6.2 Within-subject folds

Folds are made of trials, shuffled with the split seed. For liu2024 the
imagery and the rest sample of a trial are always in the same fold, so a test
sample never has its neighbour in time in the training set. For bci2a and
jia2019, with one sample per trial, the folds are stratified by class; for
bci2a the split is the one of versions 23 to 26. The 60 epochs were chosen in version 23 for bci2a
and were not re-tuned for liu2024.

### 6.3 What is used from the test subject

Without their labels, the test subject's samples are used in the alignment
and, in leave one subject out, in the logit centring and the target
calibration. These assume that the subject's samples are available as a group
before the predictions are made. The labels are used once, to score the
predictions.

## 7. Results

Score after the final epoch, seed 12345.

| Dataset | Protocol | Accuracy | Status |
|---|---|---:|---|
| liu2024, imagery against rest | Leave one subject out | **71.24%** | Measured with version 26, same configuration |
| liu2024, imagery against rest | Within subject | | Not run yet |
| bci2a, left against right | Leave one subject out | 78.47% | Measured with version 26, same configuration |
| bci2a, left against right | Within subject | | Not run yet with this configuration |
| jia2019, left against right | Leave one subject out | | Not run yet with the whole epoch and the average reference (version 26, 3.5 s after the cue, two hybrid blocks: 47.75%) |
| jia2019, left against right | Within subject | | Not valid on this dataset (4.3); version 26 gave 94.08% |

On real recordings and on the CPU, versions 26 and 27 build identical graphs
and give identical fold results for both datasets and both protocols, so the
two measured figures are the results of this code. GPU runs differ from one
another by a fraction of a point. The source files are in
`databci/bciupperlimbactionexperiment/ourexpv26mix/outputs/`:
`Version26Mix_ImageryRest_liu2024_LOSO_seed12345.json` and
`Version26Mix_Default3Blocks_bci2a_LOSO_seed12345.json`.

### Metrics in the result files and on the screen

Every run prints, per subject and as the mean over subjects: accuracy,
balanced accuracy, macro F1, Cohen's kappa, macro precision, macro recall and
the area under the ROC curve (from the ensemble's probability of class 1).

Next to the result, which is the score after the final epoch, every fold also
records the **highest-test-epoch score**: the metrics of the epoch with the
highest test accuracy. This is how PA-TCNet and many other papers report, so
it is the number to set beside theirs. It is chosen with the test labels and
is biased upward; nothing in the training uses it. A result file holds, per
fold, `highest_epoch`, `test_accuracy_per_epoch`, and the labels, predictions
and scores of the final and of the highest epoch.

Both scores are saved with the result, not only printed: in the JSON file
(`summary` and `summary.highest_epoch`, and per subject), and as a readable
table in `<result name>.summary.txt` next to it.

### What else a result records, for the manuscript

Recording these does not change the training: the predictions of version 27
are identical to those of version 26 for the same seed (checked on CPU, Liu
2024 and BCI IV 2a).

| What | Where in the JSON | Used for |
|---|---|---|
| Labels, predictions and probability of class 1 of every test sample, final and highest epoch | per fold: `labels`, `predictions`, `scores`, `highest_epoch_predictions`, `highest_epoch_scores` | ROC curves, AUC, any metric afterwards |
| Ensemble test loss and accuracy per epoch | per fold: `test_loss_per_epoch`, `test_accuracy_per_epoch` | Training curves |
| Per model: training loss and accuracy per epoch, test loss and accuracy per epoch | per fold: `history[k].train_loss`, `train_accuracy`, `test_loss`, `test_accuracy` | Training curves (over- or under-fitting) |
| Per model: share of test samples with a calibration target, and how many are right, per epoch | per fold: `history[k].calibration_labelled_fraction`, `calibration_target_accuracy` | How the target calibration behaves (the accuracy uses the labels; a diagnostic only) |
| Training time per model and per fold; peak GPU memory | per fold: `training_seconds`, `peak_gpu_memory_bytes`; overall: `cost` | Computational cost |
| Parameters (also per path), FLOPs and MACs per sample, inference time per sample (alone and in a batch), model file size | `complexity` | Model complexity table |
| Device, GPU name, package versions | `device`, `device_name`, `packages` | Methods section |
| Features of every test sample: input of the graph head's MLP (64 values) and of the temporal encoder's output layer | `<name>.embeddings.npz`, arrays `graph_S<subject>` and `temporal_S<subject>`, in the order of the subject's labels | t-SNE |

The features come from the first model of each fold after its final epoch.
In leave one subject out every sample is a test sample once, so the t-SNE
shows all samples as seen by models that never trained on their subject.

Notes on the cost figures:

- FLOPs are counted by PyTorch (`FlopCounterMode`) for one sample and one
  model. The scatter and `index_add` operations of the graph layers and the
  element-wise operations are not counted, so the figure is a lower bound.
  A decision uses the ensemble of three models: three times the cost.
- Inference time is the median of 50 forward passes of a prepared graph, on
  the training device. Filtering, alignment and graph building are not
  included.
- Training time covers all models of a fold, including scoring the test
  samples after every epoch; graph building is not included.

On a trial run (CPU, Liu 2024): 64,862 parameters per model (graph head
56,322, wide path 3,714, temporal encoder 4,826), at least 32.8 MFLOPs per
sample, a 319 KiB model file.

`analysis/figures.py` draws, for every result file given, into
`<result name>_figures/`: `roc.png` (all test samples pooled, final and
highest epoch), `training_curves.png`, `calibration.png` (if used),
`confusion_matrix.png`, `subjects.png`, `tsne.png`, and `model.txt` (the
complexity table). With several result files, `roc_comparison.png` compares
them, for example the method and its ablations. The pooled ROC curve mixes
subjects; the mean AUC over subjects is printed on the figure as well.

### liu2024, leave one subject out, in detail

| | |
|---|---:|
| Accuracy | 71.24% |
| Macro F1 | 71.01% |
| Correct samples | 2,606 of 3,660 |
| Rest taken for imagery | 644 |
| Imagery taken for rest | 410 |
| Patients at 60% or above | 41 of 46 |
| Patients at 70% or above | 29 of 46 |
| Lowest and highest patient | 49% and 89% |
| Linear classifier on the same samples (tangent space) | 69.3% |
| The same with shuffled labels | 50.1% |

Target calibration gave a target to 45% of a test patient's samples; 80% of
those targets were right.

### A note on bci2a

This version uses three hybrid blocks, the setting of the stroke result. On
bci2a two blocks measured better in version 26 (79.90% as the mean of five
seeds, 80.99% on seed 12345, against 78.47% with three). The method of this
version is kept the same for both datasets.

## 8. What the stroke result means, and what it does not

**It means:** for a stroke patient the model has not seen, it can tell from
1.9 s of EEG whether the patient is in the imagery period or in the break,
7 times out of 10. For rehabilitation, where the system mainly needs to know
that the patient is engaged in the task, this is the relevant question.

**It does not mean** that motor imagery alone is decoded. During the imagery
the patient watches a video and during the break a still picture. A linear
classifier on parts of the head gives:

| Electrodes | Accuracy |
|---|---:|
| Motor area (FC3 FC4 C3 C4 Cz CP3 CP4 FCz) | 65.7% |
| Occipital and parietal (O1 Oz O2 P3 P4 Pz T5 T6) | 65.2% |
| Frontal (FP1 FP2 F7 F8 F3 F4 Fz) | 62.7% |
| The two EOG channels alone | 61.1% |

The visual cortex and the eye channels separate the two periods almost as
well as the motor area. The 71% is therefore a mixture of imagery, visual
processing and eye activity, and should be described as "imagery period
against break".

Other points to state when reporting:

- Rest is the break after the imagery, not a separate resting recording.
- Which hand was imagined cannot be decoded from these recordings: left
  against right gives 52% to 55% with this model and with classical methods.

## 9. What is not in this version

Removed because the method does not need it. All of it is in
[version 26](../ourexpv26mix/README.md), with the measurements.

| Removed | What it gave |
|---|---|
| Left against right on Liu 2024 | 52% to 55%, at or near chance |
| Shuffled-label control, subject lists | Controls and subsets |
| Learned node features (`timed`, `mixed`), electrode readout, time attention | No gain in the full model |
| Tangent-space features, low-frequency path | No gain; the low-frequency path reads eye movements on BCI IV 2a |
| Choice of the number of hybrid blocks | Ablation (path selection and the calibration switch are back, as `--paths` and `--no-target-calibration`) |
| Screening script | Analysis with a linear classifier |

## 10. Where this version comes from

| Version | Step |
|---|---|
| 18 to 23 | The model and the honest evaluation, on BCI IV 2a: from 50% to 79% |
| 24 | The reader for Liu 2024; the duplicated recordings found and left out |
| 25 | Ideas from ASA-STGCN, alignment by paralysed side, signal check: no gain |
| 26 | Jia 2019, comparison with PA-TCNet, target calibration, work on the graph head, the imagery-against-rest task |
| 27 | Only the method of the best result for stroke rehabilitation, on Liu 2024, BCI IV 2a and Jia 2019, with both protocols |
