# M-ShallowConvNet: reproduction on our datasets

Kim S.-J., Lee D.-H., Lee S.-W. (2022). "Rethinking CNN Architecture for
Enhancing Decoding Performance of Motor Imagery-Based EEG Signals". IEEE
Access 10, 96984-96996, DOI 10.1109/ACCESS.2022.3204758. Paper:
`src/temp/Rethinking_CNN_Architecture_for_Enhancing_Decoding_Performance_of_Motor_Imagery-Based_EEG_Signals.pdf`.
Code: https://github.com/comojin1994/m-shallowconvnet (PyTorch Lightning),
copied unchanged to [original/](original/README.md).

Purpose: a baseline for our method, like [EEGNet](../eegnet/README.md). It is
trained and tested on **exactly the samples, subjects and folds of version
27**, with both of version 27's protocols.

Status: **ready to run**; only short trial runs so far.

## How to run

```bash
cd /workspaces/EEG && source .venv/bin/activate
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv28reproduce/outputs
python -u -m src.ourexpv28reproduce.run --paper mshallowconvnet --dataset bci2a --protocol loso --quiet --save-models \
  --data-dir /workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a --output-dir "$OUT"
```

`--dataset {liu2024,bci2a,jia2019}`, `--protocol {loso,within}`,
`--subjects`, `--epochs` (default 1000), `--seed` (default 42, the code's),
`--save-models` (the five validation-selected models of every fold). The
result is `<output-dir>/mshallowconvnet/<dataset>/MShallowConvNet_<dataset>_<LOSO|Within>_seed42.json`
with a `.summary.txt`; figures with
`python -m src.ourexpv28reproduce.evaluation.figures <result>.json`.

### Run time

Five models of 1,000 epochs per test fold make this the most expensive
baseline. Measured on a GTX 1060 (cuDNN benchmark on, as in the code):

| Dataset | Leave one subject out | Seconds per model and epoch |
|---|---|---|
| bci2a (22 x 500, about 800 training samples) | 31 s per fold per 40 epochs | about 0.16 |
| jia2019 (63 x 1,708, about 900 training samples) | 83 s per fold per 4 epochs | about 4.2 |

At these rates a full bci2a leave-one-subject-out run takes about 2 hours
and jia2019 several days on the GTX 1060. An RTX 3080 Ti is several times
faster (its convolutions use TF32), but jia2019 will still take the longest
of all runs. The time per subject is printed as each subject finishes.

## The three scores of every run

| Score | How it is chosen | Uses the test labels |
|---|---|---|
| **validation-selected** (main) | The paper's protocol: five models, each at its lowest validation loss, bagged by vote | No |
| **final epoch** | The five models after epoch 1,000, bagged | No |
| **highest test epoch** | The epoch at which the bagged models are most accurate on the test samples | Yes; for comparison only |

The bagged decision is the models' vote, as `evaluation.py` (one-hot
predictions averaged). The probability behind the ROC curve and the AUC is
the mean of the five models' probabilities; the vote alone gives only six
distinct values. Everything else recorded is as for EEGNet: per-sample
predictions and probabilities, training, validation and test curves (mean
over the five models) and the bagged test accuracy per epoch, the epoch each
model selected, time, memory and model size.

## 1. What the paper reports

Within subject on BCI IV 2a, **four classes**, trained on the first session
(five-fold cross-validation, five bagged models) and tested on the second
session: accuracy 0.8164, kappa 0.7551 (A1-A9: 0.910, 0.556, 0.938, 0.806,
0.816, 0.660, 0.938, 0.851, 0.875). BCI IV 2b (two classes, three
electrodes): 0.8647. These are not comparable with our two-class tasks and
our folds; this is the model and its training applied to our tasks.

The paper's settings (kernel sizes, pooling, number of kernels) were chosen
in ablation tables (Tables 2-5) that report test-session accuracy.

## 2. The specification, and where each part comes from

| Part | Value | Source |
|---|---|---|
| Model | Temporal convolution 1 x 30 (fs x 0.12), 24 kernels, 'same'; spatial convolution channels x 1, 24 kernels; square; average pool 1 x 75 (fs x 0.3), stride 22 (overlap 0.7); log (clamped at 1e-6); batch normalisation after the log (not after the spatial convolution); dropout 0.5; dense | Section II-D, Figure 2, mshallowconvnet.py |
| Max-norm | 2 on the convolutions, renormed before every forward pass | Section II-E, layers.py |
| Max-norm on the dense layer | 0.5, applied **once** (see below) | layers.py, as it behaves |
| Optimiser | Adam, learning rate 0.002, weight decay 0.075, cosine annealing to 0 over the epochs | Section II-E; weight decay from the code's config only |
| Epochs, batch | 1,000, 64 | Section II-E |
| Loss | Cross-entropy with the CutCat soft labels | Section II-C, litmodel.py |
| Augmentation | CutCat: a stretch of up to lambda x T replaced by a sample of the other class from the same mini-batch; lambda 0.125 for bci2a, 0.1 for liu2024 and jia2019 | Section II-C (0.125 for 2a; 0.1 for its two-class datasets) |
| Model selection | Five models (five-fold split of the training side), each at its lowest validation loss; bagged | Section II-E, evaluation.py |
| Pre-processing | Microvolts; low-pass 38 Hz, 4th-order Butterworth; exponential moving standardisation (alpha 0.999), each sample on its own with a 4 s initial block | Section II-B, bci_compet.py |
| Sampling rate | 250 Hz (no resampling) | Code (`downsampling: 0`) |
| Seed | 42 | Code (configs) |

The port has the same parameters as the published model (16,468 for its
four-class BCI IV 2a input; tested).

## 3. Where the code and the paper differ, and what was done

- **The dense layer's max-norm acts once.** The code's
  `LazyLinearWithConstraint` renorms in `forward`, but PyTorch turns a lazy
  layer into a plain `nn.Linear` at its first call, so the constraint is
  applied at the first forward pass only. Running the original confirms it.
  The port does the same (renorm once, after construction).
- **Low-pass filter:** the paper says a 4th-order Butterworth filter, the code
  uses MNE's FIR filter. The paper's filter is used, zero-phase as MNE's.
- **CutCat partner:** the paper takes the partner from the same mini-batch;
  the code takes it from the whole training set, including the validation
  fold's samples. The paper's version is used, so no validation sample
  enters training.
- **Model selection:** the paper keeps the lowest-validation-loss checkpoint;
  the code keeps the three best and loads the last file name in sorted order.
  The paper's rule is used.
- **Window:** the paper says 3-6 s of the trial (1-4 s after the cue), the
  code takes 0-3 s after the cue. Neither applies here: version 27's windows
  are used.

## 4. What is ours, not the paper's

| | Here (version 27's) | Paper (BCI IV 2a) |
|---|---|---|
| Task | liu2024: imagery against rest; bci2a and jia2019: left against right hand | 4 classes |
| Windows | liu2024: 1.9 s; bci2a: 0-2 s after the cue; jia2019: the whole 6.8 s epoch with a common average reference | 3 s |
| Protocol | Leave one subject out, and within subject with version 27's 10 folds | Within subject, session 1 to session 2 |
| The five-fold split | Leave one subject out: five groups of training subjects (validation by subject). Within subject: five parts of the nine training folds, trials kept together | Five folds of the training session's trials |
| CutCat lambda for liu2024 and jia2019 | 0.1, the paper's value for its two-class datasets | - |

No Euclidean alignment or other adaptation to the test subject: the paper
has none.

## 5. Files

```
settings.py      model, preparation and training settings
preparation.py   version 27's samples: microvolts, low-pass 38 Hz, standardisation per sample
network.py       M-ShallowConvNet, ported from original/model/
training.py      one test fold: CutCat, five models, validation selection, bagging, the three scores
experiment.py    leave one subject out and within subject over a dataset
command.py       the command-line options and the result name
original/        the published code, unchanged
```
