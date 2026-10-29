# EEG Conformer: reproduction on our datasets

Song Y., Zheng Q., Liu B., Gao X. (2023). "EEG Conformer: Convolutional
Transformer for EEG Decoding and Visualization". IEEE Transactions on Neural
Systems and Rehabilitation Engineering 31, 710-719. Paper:
`src/temp/EEG_Conformer_Convolutional_Transformer_for_EEG_Decoding_and_Visualization.pdf`.
Code: https://github.com/eeyhsong/EEG-conformer (PyTorch, GPL v3), copied
unchanged to [original/](original/README.md).

Purpose: a baseline for our method, like [EEGNet](../eegnet/README.md) and
[M-ShallowConvNet](../mshallowconvnet/README.md). It is trained and tested on
**exactly the samples, subjects and folds of version 27**, with both of
version 27's protocols.

Status: **ready to run**; only short trial runs so far.

## How to run

```bash
cd /workspaces/EEG && source .venv/bin/activate
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv28reproduce/outputs
python -u -m src.ourexpv28reproduce.run --paper conformer --dataset bci2a --protocol loso --quiet --save-models \
  --data-dir /workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a --output-dir "$OUT"
```

`--dataset {liu2024,bci2a,jia2019}`, `--protocol {loso,within}`,
`--subjects`, `--epochs` (default 2000, the code's), `--seed` (default
12345), `--save-models` (the model after the last epoch of every fold). The
result is `<output-dir>/conformer/<dataset>/Conformer_<dataset>_<LOSO|Within>_seed12345.json`
with a `.summary.txt`; figures with
`python -m src.ourexpv28reproduce.evaluation.figures <result>.json`.

### Run time and memory

2,000 epochs per fold, measured on a GTX 1060:

| Run | Seconds per epoch | One fold (2,000 epochs) | Whole run |
|---|---:|---:|---:|
| bci2a, leave one subject out | 0.9 | about 30 min | about 4 h |
| liu2024, leave one subject out | 3.5 | about 2 h | about 90 h |
| liu2024, within subject | 0.07 | about 2.5 min | about 18 h |

An RTX 3080 Ti is several times faster, but liu2024 leave one subject out
remains a run of a day or more. jia2019 (63 x 1,708 samples) needs about 5 GB
of GPU memory per training step (34 MB per sample, batches of 72 real plus 72
augmented samples): it does not fit on the 6 GB GTX 1060 and fits on a 12 GB
card. The time per subject is printed as each subject finishes.

## The two scores of every run

| Score | How it is chosen | Uses the test labels |
|---|---|---|
| **final epoch** (main) | After epoch 2,000 | No |
| **highest test epoch** | The published code's score ("The best accuracy"): the epoch with the highest test accuracy | Yes; for comparison with the paper only |

The code has no validation set, so there is no label-free selection other
than the final epoch. Its second printed number, the test accuracy averaged
over all epochs, is kept per fold as `mean_test_accuracy_over_epochs`. As for
the other baselines, every test sample's prediction and probability (ROC),
the training and test curves, the time, memory and model size are recorded.

## 1. What the paper reports

Within subject: BCI IV 2a (four classes, session 1 for training, session 2
for testing, 4 s from the cue), BCI IV 2b (two classes, three electrodes) and
SEED (emotions, five-fold). The accuracies are in Tables II-IV, which are
images in our copy of the PDF. They are not comparable with our two-class
tasks and protocols; this is the model and its training applied to our tasks.
Given the code, the reported accuracy is most likely the highest test
accuracy over the epochs (see above).

## 2. The specification, and where each part comes from

| Part | Value | Source |
|---|---|---|
| Convolution module | Conv 1 x 25 (40 kernels), Conv channels x 1 (40), BatchNorm, ELU, AvgPool 1 x 75 stride 15, Dropout 0.5, Conv 1 x 1; the pooled time steps are the tokens | Section II-C, Table I, conformer.py |
| Self-attention | 6 layers, 10 heads, token size 40; scores divided by the square root of 40; feed-forward 160 with GELU; dropout 0.5; pre-LayerNorm, residual | Section II-C, Section III-C, conformer.py |
| Classifier | Linear 256, ELU, Dropout 0.5, Linear 32, ELU, Dropout 0.3, Linear (classes) | conformer.py (the paper says "two fully-connected layers") |
| Optimiser | Adam, learning rate 0.0002, betas 0.5 and 0.999 | Section III-C |
| Epochs, batch | 2,000; 72 real samples plus 72 S&R samples per step | conformer.py; the paper shows 2,000 epochs (Fig. 2) |
| Augmentation | S&R: Ns = 8 time segments from random training samples of the same class, in time order; as many new samples as the batch, split equally over the classes | Section III-B, conformer.py |
| Band-pass | Chebyshev type II, order 6, 60 dB, 4-40 Hz, zero-phase, on every cut-out sample | Section II-B, preprocessing/BCIIV2a.m |
| Standardisation | z-score with the mean and standard deviation of all training values, applied to the test samples | Section II-B, conformer.py |
| Sampling rate | 250 Hz | Code |
| Seed | 12345 | The code draws a random seed per subject; ours |

The port computes the same function as the published model: with the
published model's weights copied in, the outputs are identical (tested; the
code's unused `clshead` is left out). The spatial convolution is computed as
a tensor product, which gives the same result as `nn.Conv2d` and its
gradients (tested), runs about four times faster, and avoids the
multi-gigabyte cuDNN workspace that `nn.Conv2d` takes on its first call.

## 3. What is ours, not the paper's

| | Here (version 27's) | Paper (BCI IV 2a) |
|---|---|---|
| Task | liu2024: imagery against rest; bci2a and jia2019: left against right hand | 4 classes |
| Windows | liu2024: 1.9 s; bci2a: 0-2 s after the cue; jia2019: the whole 6.8 s epoch with a common average reference | 0-4 s after the cue |
| Protocol | Leave one subject out (train on all other subjects), and within subject with version 27's 10 folds | Within subject, session 1 to session 2 |
| Classifier output | 2 classes | 4 (fixed in the code) |
| Filtering | Every sample filtered on its own, as the MATLAB code filters its cut-out trials; second-order sections instead of the MATLAB transfer function, for numerical stability | MATLAB `filtfilt` |

The sample is filtered on its own because the published pre-processing does
so; with liu2024's 1.9 s samples the filter's edges take up a larger share of
the sample than with the paper's 4 s trials.

No Euclidean alignment or other adaptation to the test subject: the paper
has none.

## 4. Files

```
settings.py      model, preparation and training settings
preparation.py   version 27's samples, band-passed; the z-score per fold
network.py       EEG Conformer, ported from original/conformer.py
training.py      one fold: S&R, 2,000 epochs, the two scores, curves, time, memory
experiment.py    leave one subject out and within subject over a dataset
command.py       the command-line options and the result name
original/        the published code (conformer.py, the BCI IV 2b script, MATLAB pre-processing), unchanged
```
