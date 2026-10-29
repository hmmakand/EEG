# UA-DAAN: reproduction on our datasets

Shen J., You L., Ma Y., Zhao Z., Liang H., Zhang Y., Hu B. (2025).
"UA-DAAN: An Uncertainty-Aware Dynamic Adversarial Adaptation Network for
EEG-Based Depression Recognition". IEEE Transactions on Affective Computing
16(3), 2130-2141, DOI 10.1109/TAFFC.2025.3555433. Paper:
`src/temp/UA-DAAN_An_Uncertainty-Aware_Dynamic_Adversarial_Adaptation_Network_for_EEG-Based_Depression_Recognition.pdf`.

**No code from the authors.** The paper publishes none. The implementation
followed is a third-party one, https://github.com/chenjlbj/EEG-Depression-MSUA-DAAN
(UA-DAAN and its multi-source extension), copied to
[original/](original/README.md). The DBConformer repository
(https://github.com/wzwvv/DBConformer) contains no UA-DAAN model, only generic
adversarial parts (gradient reversal, discriminators). The reference matches
the paper's description (Section III and IV-C) and fills in what the paper
leaves out (layer sizes, prior, Monte Carlo samples, loss scales).

Purpose: a baseline for our method. **UA-DAAN was designed for depression
recognition** from resting-state EEG (differential-entropy features, depressed
against healthy subjects); it is applied here unchanged to our tasks. It is
trained and tested on **exactly the subjects, tasks, samples and test folds
of version 27**, with both of version 27's protocols.

Status: **ready to run**; only short trial runs so far.

## How to run

```bash
cd /workspaces/EEG && source .venv/bin/activate
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv28reproduce/outputs
python -u -m src.ourexpv28reproduce.run --paper uadaan --dataset bci2a --protocol loso --quiet --save-models \
  --data-dir /workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a --output-dir "$OUT"
```

`--dataset {liu2024,bci2a,jia2019}`, `--protocol {loso,within}`,
`--subjects`, `--epochs` (default 30), `--seed` (default 1234, the
reference's), `--save-models` (the model after the last epoch of every fold).
The result is `<output-dir>/uadaan/<dataset>/UADAAN_<dataset>_<LOSO|Within>_seed1234.json`
with a `.summary.txt`; figures with
`python -m src.ourexpv28reproduce.evaluation.figures <result>.json`.

Run time on a GTX 1060: about 13 s per bci2a leave-one-subject-out fold and
1.5 s per liu2024 within-subject fold; whole runs take minutes to about an
hour.

## The two scores of every run

| Score | How it is chosen | Uses the test labels |
|---|---|---|
| **final epoch** (main) | After epoch 30, as the reference scores it; version 27's protocol | No |
| **highest test epoch** | The epoch with the highest test accuracy | Yes; for comparison only |

**UA-DAAN is transductive**: the test samples, without their labels, are the
target domain during training. Our method also uses the test subject's
unlabelled samples (alignment, logit centring, target calibration); EEGNet,
M-ShallowConvNet, EEG Conformer and MSCFormer do not.

Every prediction is the mean of 10 Monte Carlo passes (dropout on, Bayesian
weights sampled), as in the reference. Recorded per fold, besides the usual
predictions, probabilities, time, memory and size: the classification and
transfer losses, the training accuracy and the test loss and accuracy per
epoch, and the final dynamic adversarial factor.

## 1. What the paper reports

Depression recognition on three datasets (MODMA, 53 subjects, 128 channels;
two 3-channel datasets with 170 and 214 subjects), subject-independent
10-fold cross-validation, differential-entropy features (Table I, an image in
our copy of the PDF). Not comparable with motor imagery.

## 2. The specification, and where each part comes from

| Part | Value | Source |
|---|---|---|
| Input | Differential entropy per channel in five bands (delta, theta, alpha, beta, gamma), after a 1-40 Hz band-pass | Section IV-B, IV-D |
| Feature extractor | BatchNorm, Linear 256, Dropout 0.2, ReLU, Linear 64, Dropout, ReLU, Linear 16, Dropout; MC Dropout (also at test time) | Section III-B, IV-C; extractor.py |
| Label classifier | Bayesian linear layers 16 to 4 (Tanh) to 2, BatchNorm, Softmax; prior N(0, 0.1^2); loss 2 x cross-entropy + 0.01 x KL to the prior | Section III-C, IV-C; classifier.py, model.py |
| Discriminators | One global and one local per class (input p_c x f), 16 to 4 (ReLU) to 2, Softmax, behind gradient reversal (DANN schedule, gamma 1, 1,000 steps) | Section III-D; discriminator.py |
| Domain loss | Entropy of the discriminator's prediction (Eq. 10), weighted by 1 + exp(-H(p)) with H(p) the entropy of the class prediction (Eq. 9); global x 0.1, local x 0.05 | Eq. 9-12; discriminator.py |
| Dynamic factor | From the global and local A-distances of the previous epoch (Eq. 13-15) | Eq. 13-15; discriminator.py |
| Total loss | L_y - ((1 - omega) L_g + omega L_l) (lambda = 1) | Eq. 16; ua-daan12fold.py |
| Monte Carlo | 10 passes, averaged, in training and prediction | model.py |
| Training | Adam, learning rate 0.001, weight decay 0.001, source batch 98, target batch 8, 30 epochs | Section IV-C; scripts |
| Seed | 1234 | ua-daan12fold.py |

The port gives the same losses, gradients and dynamic factor as the reference
code with the same weights and random numbers (tested over several steps).

## 3. Choices where the paper and the reference are silent or differ

- **Training loop:** the reference's `main.py` takes only as many steps per
  epoch as the target has batches of 8 (with our within-subject test folds of
  8 samples: one step per epoch); its `ua-daan12fold.py` takes one step per
  source batch and cycles the target batches. The latter is used.
- **Evaluation:** MC Dropout at test time (dropout on, batch normalisation in
  evaluation mode), as the reference's `main.py` and the paper's Section
  III-B; `ua-daan12fold.py` switches dropout off.
- **Band edges:** not given in the paper: delta 1-4, theta 4-8, alpha 8-13,
  beta 13-30, gamma 30-40 Hz (inside the paper's 1-40 Hz band-pass), 4th-order
  zero-phase Butterworth filters.
- **Validation and grid search:** the paper tuned parameters on a validation
  tenth of the training subjects; the reference uses fixed values and scores
  the last epoch, which is done here.
- **Batches of one sample** are skipped (batch normalisation cannot train on
  them).
- Kept from the reference although unusual: the cross-entropy is taken on the
  averaged probabilities (a softmax before PyTorch's cross-entropy), and the
  domain loss does not use the domain labels (it is the entropy of Eq. 10).

## 4. What is ours, not the paper's

| | Here (version 27's) | Paper |
|---|---|---|
| Task | liu2024: imagery against rest; bci2a and jia2019: left against right hand | Depressed against healthy |
| Input | DE of version 27's samples: liu2024 29 x 5 = 145 values, bci2a 22 x 5 = 110, jia2019 63 x 5 = 315 | DE of 10 s (or 6 s) resting-state segments; 128 x 5 = 640 for MODMA |
| Target domain | Leave one subject out: the held-out subject. Within subject: version 27's test fold | The test subjects of a subject-independent fold |
| Units | Volts; the BCI IV 2a reader returns microvolts, converted first | - |

## 5. Files

```
settings.py      feature, model and training settings
features.py      version 27's samples -> differential entropy per channel and band
network.py       UA-DAAN, ported from the reference code
training.py      one fold: source and target batches, 30 epochs, MC evaluation, the two scores
experiment.py    leave one subject out and within subject
command.py       the command-line options and the result name
original/        the reference implementation, unchanged
```
