# MSCFormer: reproduction on our datasets

Zhao W., Zhang B., Zhou H., Wei D., Huang C., Lan Q. (2025). "Multi-scale
convolutional transformer network for motor imagery brain-computer
interface". Scientific Reports 15, 12935, DOI 10.1038/s41598-025-96611-5.
Paper: `src/temp/Multi-scale convolutional transformer network for motor imagery brain-computer interface.pdf`.
Code: https://github.com/snailpt/MSCFormer (PyTorch, Apache 2.0), copied
unchanged to [original/](original/README.md).

Purpose: a baseline for our method, like the other papers here. It is
trained and tested on **exactly the subjects, tasks, samples and test folds
of version 27**, with both of version 27's protocols.

Status: **ready to run**; only short trial runs so far.

## How to run

```bash
cd /workspaces/EEG && source .venv/bin/activate
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv28reproduce/outputs
python -u -m src.ourexpv28reproduce.run --paper mscformer --dataset bci2a --protocol loso --quiet --save-models \
  --data-dir /workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a --output-dir "$OUT"
```

`--dataset {liu2024,bci2a,jia2019}`, `--protocol {loso,within}`,
`--subjects`, `--epochs` (default 1000), `--models` (default 5), `--seed`
(default 12345), `--save-models` (the five validation-selected models of
every test fold). The result is
`<output-dir>/mscformer/<dataset>/MSCFormer_<dataset>_<LOSO|Within>_seed12345.json`
with a `.summary.txt`; figures with
`python -m src.ourexpv28reproduce.evaluation.figures <result>.json`.

### Run time

Five models of 1,000 epochs per test fold, each step 288 samples (72 real,
216 augmented). Measured on a GTX 1060:

| Run | Per fold and epoch (5 models) | One fold, 1,000 epochs | Whole run |
|---|---:|---:|---:|
| bci2a, leave one subject out (8 folds) | 12 s | about 3.5 h | about 28 h |
| liu2024, within subject (460 folds) | 1.3 s | about 21 min | about 7 days |
| liu2024, leave one subject out (46 folds) | 55 s | about 15 h | about 4 weeks |

An RTX 3080 Ti is several times faster, but the full liu2024 runs remain far
longer than any other model here. Fewer epochs (`--epochs`) or models
(`--models`), fixed in advance, shorten them proportionally; the result file
records the values used. The time per subject is printed as each finishes.

## The three scores of every run

| Score | How it is chosen | Uses the test labels |
|---|---|---|
| **validation-selected** (main) | The paper's protocol: each of the five models at its lowest validation loss; their test metrics averaged | No |
| **final epoch** | Each model after epoch 1,000; metrics averaged | No |
| **highest test epoch** | Each model at the epoch with its highest test accuracy; metrics averaged | Yes; for comparison only |

The paper averages the five models' results; it does not combine their
predictions. A subject's metrics are therefore the mean over the five models
of each model's metrics on all the subject's test samples (within subject:
over its 10 test folds); `accuracy_per_model` lists the five accuracies. For
the ROC curves the five models' mean probability is stored as
`<score>_scores`, with its own metrics under `<score>_mean_probability`.

## 1. What the paper reports

Within subject, BCI IV 2a (four classes) and 2b (two classes), training
session to test session, five-fold split of the training session: 82.95%
(kappa 0.7726) on 2a and 88.00% (kappa 0.7599) on 2b. These are not
comparable with our two-class tasks and folds; this is the model and its
training applied to our tasks.

## 2. The specification, and where each part comes from

| Part | Value | Source |
|---|---|---|
| Convolution module | Three branches, temporal kernels 85, 65, 45 (16 filters, 'same'); depthwise spatial convolution (channels x 1); BatchNorm; ELU; average pool (1, 44); dropout | Paper (Fig. 2, Table 2), MSCFormer_model.py |
| Tokens | The three branches concatenated: 48 features per pooled time step; a class token in front; multiplied by sqrt(48); learned position embedding (100 places), dropout 0.1 | Paper (Eq. 2), code |
| Transformer | 5 layers, 8 heads, post-LayerNorm residuals, feed-forward 192 with GELU, dropout 0.5; scores divided by sqrt(48) | Paper (Table 2), code |
| Classifier | The class token's output, dropout 0.25, one linear layer | Paper, code |
| Branch dropout | 0.5 within subject (paper); 0.25 across subjects (code: "0.25 for cross-subject") | Paper, code |
| Pooling size | 44 (the paper's best for BCI IV 2a; 52 was for 2b's three electrodes); also used for liu2024 and jia2019 | Paper (Fig. 8), code |
| Pre-processing | None ("we did not apply any band-pass filtering or artifact removal"); z-score with one mean and one standard deviation of the training data | Paper, code |
| Augmentation | S&R, 8 segments, three times as many augmented samples as real ones in every batch | Paper, code |
| Optimiser | Adam, learning rate 0.001, betas 0.5 and 0.999; weight decay 0 (the paper adds 0.001 only for 2b's three electrodes; our datasets have 22 to 63) | Paper, code |
| Epochs, batch | 1,000; 288 (72 real + 216 augmented) | Paper, code |
| Model selection | Five-fold split of the training side; lowest validation loss; test results averaged | Paper |
| Seed | 12345 | The code draws a random seed per subject; ours |

The port computes the same function as the published model: with the
published weights copied in, the outputs are identical (tested). The
depthwise spatial convolution is computed as a tensor product, which gives
the same result and gradients as the grouped `nn.Conv2d` (tested) and makes
training about 25% faster.

## 3. Where the code and the paper differ, and what was done

- **Model selection:** the paper keeps the lowest validation loss; the code
  keeps the highest validation accuracy (ties by lower loss). The paper's
  rule is used.
- **S&R source:** the code draws the augmentation segments from all training
  samples, including the validation part; the paper combines "the remaining
  four subsets with the S&R augmented dataset". Here only the model's own
  training parts are used, so no validation sample enters training.
- **Class token:** the paper calls it learnable; the code prepends zeros. With
  the learned position embedding added, both give a learned first token; the
  code's form is kept.
- **Window:** the paper uses 2-6 s (BCI IV 2a). Version 27's windows are used.

## 4. What is ours, not the paper's

| | Here (version 27's) | Paper |
|---|---|---|
| Task | liu2024: imagery against rest; bci2a and jia2019: left against right hand | 4 classes (2a), 2 classes (2b) |
| Windows | liu2024: 1.9 s (10 tokens); bci2a: 0-2 s after the cue (11 tokens); jia2019: the whole 6.8 s epoch with a common average reference (38 tokens) | 4 s (22 tokens) |
| Protocol | Leave one subject out, and within subject with version 27's 10 folds | Within subject, training session to test session |
| The five-fold split | Leave one subject out: five groups of training subjects. Within subject: five parts of the nine training folds, trials kept together | Five parts of the training session, stratified by class |

No Euclidean alignment or other adaptation to the test subject: the paper
has none.

## 5. Files

```
settings.py      model and training settings
preparation.py   version 27's samples, unfiltered; the z-score
network.py       MSCFormer, ported from original/MSCFormer_model.py
training.py      one test fold: five models, S&R, validation selection, the three scores
experiment.py    leave one subject out and within subject; the averaging over the five models
command.py       the command-line options and the result name
original/        the published code, unchanged
```
