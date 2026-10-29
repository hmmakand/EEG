# EEGNet: reproduction on our datasets

Lawhern V. J., Solon A. J., Waytowich N. R., Gordon S. M., Hung C. P.,
Lance B. J. (2018). "EEGNet: a compact convolutional neural network for
EEG-based brain-computer interfaces". J. Neural Eng. 15, 056013.
Paper: `src/temp/EEGNet: A Compact Convolutional Neural Network for EEG-based
Brain-Computer Interfaces.pdf`. Code: https://github.com/vlawhern/arl-eegmodels
(Keras), copied unchanged to [original/](original/README.md).

Purpose: a baseline for our method. EEGNet is trained and tested on **exactly
the samples, subjects and folds of version 27**, with both of version 27's
protocols, so that every test sample is scored by both methods.

Status: **ready to run**; only short trial runs so far.

## How to run

```bash
cd /workspaces/EEG && source .venv/bin/activate
LIU=/workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data
BCI=/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a
JIA=/workspaces/datadisk/databci/eegdata/jia2019
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv28reproduce/outputs

for DATASET in liu2024 bci2a jia2019; do
  case $DATASET in liu2024) DIR=$LIU;; bci2a) DIR=$BCI;; jia2019) DIR=$JIA;; esac
  python -u -m src.ourexpv28reproduce.run --paper eegnet --dataset $DATASET --protocol loso   --quiet --save-models --data-dir "$DIR" --output-dir "$OUT"
  python -u -m src.ourexpv28reproduce.run --paper eegnet --dataset $DATASET --protocol within --quiet --save-models --data-dir "$DIR" --output-dir "$OUT"
done
python -m src.ourexpv28reproduce.evaluation.figures "$OUT"/eegnet/*/EEGNet_*.json
```

Each run writes `<output-dir>/eegnet/<dataset>/EEGNet_<dataset>_<LOSO|Within>_seed12345.json`,
a readable `.summary.txt`, and with `--save-models` the validation-selected
model of every fold in `models/<result name>/`. Options: `--subjects`,
`--epochs` (default 500), `--seed` (default 12345), `--quiet`.

On a GTX 1060 a BCI IV 2a leave-one-subject-out fold takes about 3.5 s per
30 epochs (about 1 minute for 500 epochs); a Liu 2024 fold has about four
times as many training samples.

## The three scores of every run

| Score | How it is chosen | Uses the test labels |
|---|---|---|
| **validation-selected** (main) | The paper's protocol: 500 epochs, keeping the weights with the lowest validation loss. The validation set is taken from the training side of the fold | No |
| **final epoch** | After epoch 500, as version 27 scores our method | No |
| **highest test epoch** | The epoch with the highest test accuracy | Yes; for comparison only |

Accuracy, balanced accuracy, macro F1, Cohen's kappa, macro precision, macro
recall and AUC are recorded for all three, with every test sample's
prediction and probability (ROC curves), the loss and accuracy per epoch for
training, validation and test, the training time, peak GPU memory, and the
model's size and cost (`complexity`), as for PA-TCNet.

## 1. What the paper reports

EEGNet-8,2 is compared with DeepConvNet, ShallowConvNet and FBCSP on four
data sets. On BCI IV 2a (called SMR in the paper) it classifies all **four**
classes, within subject (four-fold blockwise) and across subjects (5 training
and 3 validation subjects, the competition's test session of the held-out
subject). The results are given only as bar charts (Figures 3 and 5), without
numbers. They are not comparable with our two-class tasks; the reproduction
here is the model and its training applied to our tasks, not a replication of
the paper's numbers.

The one number that can be checked exactly is the model size: Table 3 gives
1,716 trainable parameters for EEGNet-8,2 on the SMR data (22 channels, 2 s
at 128 Hz, 4 classes, kernel 32). The port has exactly 1,716 (tested).

## 2. The specification, and where each part comes from

| Part | Value | Source |
|---|---|---|
| Model | EEGNet-8,2: F1 = 8, D = 2, F2 = 16; temporal kernel (1, 32); depthwise spatial filter (C, 1) with max norm 1; average pool (1, 4); separable convolution (1, 16); average pool (1, 8); dense with max norm 0.25; ELU; batch normalisation; no bias in the convolutions | Table 2, EEGModels.py |
| Temporal kernel | 32 samples instead of 64, "since the data were high-passed at 4 Hz" | Section 2.3 (SMR) |
| Dropout | 0.25 across subjects, 0.5 within subject | Section 2.2.1 |
| Optimiser | Adam, Keras defaults (learning rate 0.001, epsilon 1e-7) | Section 2.2.1 |
| Loss | Categorical cross-entropy | Section 2.2.1 |
| Epochs | 500, weights with the lowest validation loss | Section 2.2.1 |
| Class weights | Inverse proportion, majority class 1, rounded up; only for unequal classes | Section 2.3 |
| Sampling rate | 128 Hz | Section 2.1.4 |
| Band | 4-40 Hz | Table 1 |
| Pre-processing | As Schirrmeister et al. 2017 [32], with their software (braindecode): microvolts, causal Butterworth band-pass, exponential moving standardisation (factor 0.001, first 1,000 samples standardised as a block, eps 1e-4) | Section 2.1.4 |
| Batch size | 16 | Not in the paper; the code's example (examples/ERP.py) |
| Cross-subject validation set | 3/8 of the other subjects, at random (seeded) | The paper's SMR split, 3 of the 8 other subjects |
| Seed | 12345 | Not in the paper; our seed |

The port follows the Keras code where PyTorch's defaults differ (batch-norm
momentum and epsilon, Glorot initialisation with Keras's fan-in and fan-out,
'same' padding, max-norm constraints after every step); see
[network.py](network.py).

## 3. What is ours, not the paper's

These follow version 27, so that the comparison is on the same samples:

| | Here (version 27's) | Paper (SMR) |
|---|---|---|
| Task | liu2024: imagery against rest; bci2a and jia2019: left against right hand | 4 classes |
| Windows | liu2024: 1.9 s (imagery from 1 s after its onset; rest from 4 s); bci2a: 0-2 s after the cue; jia2019: the whole 6.8 s epoch with a common average reference | 0.5-2.5 s after the cue |
| Subjects | liu2024: 46; bci2a: 8 (without 4); jia2019: 15 | 9 |
| Data of the held-out subject (LOSO) | All its samples | The competition's evaluation session |
| Cross-subject folds | Every subject held out once (one repetition) | 10 random splits per subject (90 folds) |
| Within-subject folds | Version 27's 10 folds; test fold k, validation fold k + 1, training the other 8 | 4 blocks: 2 training, 1 validation, 1 test |

Assumptions where the paper is not specific:

- **Filter order 3** and **upper edge 40 Hz.** The braindecode example for
  BCI IV 2a uses a causal Butterworth filter of order 3 and 4-38 Hz; the
  paper's Table 1 says 4-40 Hz, which is used.
- **Standardisation per trial.** [32]'s code standardises continuous
  recordings. Liu 2024 and Jia 2019 are published as separate epochs, so
  every trial's whole stretch (8 s, 4 s and 6.8 s) is filtered and
  standardised on its own before the sample is cut out. With BCI IV 2a's
  4 s trials (1,000 samples) this amounts to standardising each trial with
  its own mean and standard deviation.
- **Resampling** with `scipy.signal.resample_poly` (250 to 128 Hz) after
  filtering and standardisation. Window lengths at 128 Hz: liu2024 243,
  bci2a 256, jia2019 874 samples.
- **No Euclidean alignment** and no other adaptation to the test subject:
  the paper has none. Our method uses alignment; this is part of the
  difference between the methods.

## 4. Files

```
settings.py      the settings above: model (Table 2), preparation, training
preparation.py   version 27's samples, prepared as the paper: microvolts, filter, standardisation, 128 Hz
network.py       EEGNet in PyTorch, ported from original/EEGModels.py
training.py      one fold: 500 epochs, validation stopping, the three scores, curves, time, memory
experiment.py    leave one subject out and within subject over a dataset
command.py       the command-line options, result name and summary table
original/        EEGModels.py and its licence, unchanged
```

The samples, subjects and folds are in
[../../datascripts/version27.py](../../datascripts/version27.py); on the real
recordings they were checked to be identical to version 27's (windows,
labels, and the within-subject folds).
