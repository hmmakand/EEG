# PA-TCNet: reproduction

Wang X., Zhao Y., He D., Xia Q., Li G., Wang N., Peng N., Jiang B. (2026).
"PA-TCNet: Pathology-Aware Temporal Calibration with Physiology-Guided Target
Refinement for Cross-Subject Motor Imagery EEG Decoding in Stroke Patients".
arXiv:2604.16554. Code: `src/temp/PA-TCNet-main` (model, PGTC, ROI templates,
training loop; no data loaders, no settings file, no utilities).

Status: **ready to run** (specification, data preparation, model, training
and scoring). No full run has been made yet.

## How to run

```bash
cd /workspaces/EEG && source .venv/bin/activate
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv28reproduce/outputs

python -u -m src.ourexpv28reproduce.run --paper patcnet --dataset jia2019 --quiet \
  --data-dir /workspaces/datadisk/databci/eegdata/jia2019 --output-dir "$OUT"
python -u -m src.ourexpv28reproduce.run --paper patcnet --dataset liu2024 --quiet \
  --data-dir /workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data --output-dir "$OUT"
python -u -m src.ourexpv28reproduce.run --paper patcnet --dataset liu2024 --cohort distinct --quiet \
  --data-dir /workspaces/datadisk/databci/eegdata/moabb/MNE-liu2024-data --output-dir "$OUT"
```

`--cohort published` (the default) is the paper's 24 subjects, including the
duplicated pair 47/49; `--cohort distinct` leaves out 49. Each run writes
`<output-dir>/patcnet/<dataset>/PATCNet_<dataset>_<cohort>_LOSO_seed2.json` and
a readable `.summary.txt` next to it. On a GTX 1060 a Liu 2024 fold takes about
16 minutes (200 epochs), so a Liu 2024 run takes about 6 hours.

### The two scores of every run

Every fold trains for all 200 epochs and evaluates the held-out subject after
each epoch. Two scores are computed from the same run:

| Score | How it is chosen | Uses the test labels |
|---|---|---|
| **as published** | The published code stops when the held-out subject's loss has not improved for 30 epochs, and reports the epoch with the highest accuracy on the held-out subject up to then | Yes |
| **final epoch** | The score after epoch 200 | No |

Training is the same either way (stopping only shortens it), so one run gives
both exactly. The "as published" score is the one to set beside the paper's
numbers; the "final epoch" score is the honest one. Accuracy, balanced
accuracy, macro F1, Cohen's kappa, macro precision, macro recall and AUC are
recorded for both.

### What else a result records

Recording these does not change the training (checked: a fold gives the same
per-epoch accuracies and losses before and after they were added).

| What | Where in the JSON | Used for |
|---|---|---|
| Labels, predictions and probability of class 1 of every test trial, for both scores | per fold: `labels`, `final_epoch_predictions`, `final_epoch_scores`, `as_published_predictions`, `as_published_scores` | ROC curves |
| Loss and accuracy on the source batches per epoch (training mode) | per fold: `train_loss_per_epoch`, `train_accuracy_per_epoch` | Training curves |
| Loss and accuracy on the held-out subject per epoch | per fold: `test_loss_per_epoch`, `test_accuracy_per_epoch` | Training curves |
| Share of target trials with a PGTC target, and how many are right, per epoch | per fold: `calibrated_fraction_per_epoch`, `target_accuracy_per_epoch` | How PGTC behaves (the accuracy uses the labels; a diagnostic only) |
| Training time and peak GPU memory per fold | per fold: `training_seconds`, `peak_gpu_memory_bytes`; overall: `cost` | Computational cost |
| Parameters, FLOPs and MACs per trial, inference time per trial (alone and in a batch), model file size | `complexity` | Model complexity table |
| Device, GPU name, CUDA, Python and package versions | `device`, `device_name`, `cuda`, `cudnn`, `python`, `packages` | Methods section |

The training time covers all 200 epochs; the published code would stop at
`as_published.stopping_epoch`, which can be earlier. Measured on a GTX 1060
for Liu 2024: 71,132 parameters, 56.2 MFLOPs per trial (element-wise
operations not counted), a 328 KiB model file, 7.4 ms for one trial alone
and 0.40 ms per trial in a batch of 40, peak GPU memory 1.4 GiB in training.

`python -m src.ourexpv28reproduce.evaluation.figures RESULT.json ...` draws
into `<result name>_figures/`: `roc.png` (both scores), `training_curves.png`,
`calibration.png`, `confusion_matrix.png`, `subjects.png` and `model.txt`;
with several files also `roc_comparison.png`.

## 1. What the paper reports

Leave one subject out only; no within-subject experiment. Accuracy, Cohen's
kappa, recall, precision, F1 (and AUC in the tables), averaged over the
leave-one-subject-out rounds.

| Dataset (paper's name) | Our reader | Subjects | Input | PA-TCNet | EEGNet | ShallowConvNet |
|---|---|---:|---|---:|---:|---:|
| XW-Stroke | liu2024 | 24 | 30 x 1000 | 66.56% | 61.56% | 62.29% |
| 2019-Stroke | jia2019 | 15 | 63 x 1708 | 72.75% | 55.83% | 57.25% |

## 2. The specification, and where each part comes from

| Part | Value | Source |
|---|---|---|
| Liu 2024 subjects | 2, 5, 8, 9, 11, 12, 14, 17, 21, 23, 24, 26, 27, 28, 30, 32, 33, 37, 38, 43, 44, 47, 49, 50 | Paper 3.1 and code (`DEFAULT_XW_SUB_LIST`) |
| Jia 2019 subjects | all 15 | Paper 3.1 |
| Band-pass | 8-30 Hz | Paper 3.2 |
| Sample rate | 250 Hz | Paper 3.2 |
| Reference | Common average | Paper 3.2 |
| Baseline correction | Yes | Paper 3.2 |
| ICA | Yes | Paper 3.2 |
| Input shape | 30 x 1000 (Liu), 63 x 1708 (Jia) | Paper, Table 1 |
| Model | Embedding 30, depth 2, 10 temporal filters per branch, spatial multiplier 3, kernels {36, 24, 18}, pooling 8 | Paper, Table 2 |
| Dropout | 0.5 | Code default |
| ROI | left {FC3, C3, CP3}, middle {FCz, Cz, CPz}, right {FC4, C4, CP4} | Paper 2.3 |
| PGTC settings | alpha 0.98 / 0.95, tau 0.60, delta_min 0.50 / 0.45, warm-up 25 / 10 (Liu / Jia) | Paper, Table 3 |
| Optimiser | Adam, learning rate 0.001, weight decay 0.001, at most 200 epochs | Paper 3.4 |
| Batch size | 64 | Code default |
| Early stopping | Patience 30 on the loss of the held-out subject | Code |
| Reported model | The checkpoint with the highest accuracy on the held-out subject | Code (`BestModelSaver(monitor="acc")`, reloaded before the final evaluation) |
| Seed | 2 | Code default |

The code's own defaults for PGTC (alpha 0.95, tau 0.90, delta_min 0.70,
warm-up 10) differ from Table 3; the paper's values are used.

## 3. Gaps, and what is assumed

| Gap | Assumption made here | Effect if wrong |
|---|---|---|
| The 30 Liu 2024 channels | The 29 EEG electrodes and the reference CPz (all zeros in the files). CPz is in the paper's middle ROI, and the files have exactly these 30. | Small |
| The 1000-sample Liu 2024 window | 4 s from each trial's imagery onset (the whole imagery) | Moderate |
| The 1708-sample Jia 2019 window | The whole published epoch, 0 to 6.8 s | Small (the shape leaves no other choice) |
| Filter type and order | Zero-phase Butterworth, order 4, on the whole epoch before cropping | Small |
| Order of filter and down-sampling | Readers down-sample first; the signals hold nothing above 40 Hz, so the order does not matter | None |
| Baseline correction | The mean of every channel over the analysed stretch is removed | Small |
| ICA | Not applied: the components removed are not described | Unknown; can be added as an option |
| Units | Microvolts | Can matter: the first nonlinearity of the model acts on the raw scale |
| Data loader, sample order | Trials in recording order, as our readers give them | Small |

## 4. Decisions taken

The data are used as in version 27, with the paper's guidance where it gives
one:

1. **Channel names of Jia 2019:** none are assumed (version 27 has none).
   PGTC runs on Jia 2019 with the confidence check alone, which is
   `apply_confidence_calibration` of the published `pgtc.py`. The ROI check
   needs FC3, C3, CP3, FCz, Cz, CPz, FC4, C4 and CP4, and the files do not say
   which electrodes these are.
2. **ICA:** not applied (version 27 does not; the paper does not say which
   components were removed).
3. **Scoring:** both the published protocol and the final epoch, from one run
   (see "The two scores of every run").
4. **Liu 2024 channels:** the paper's 30 (version 27's 29 electrodes and the
   reference CPz, which is in the paper's middle ROI and in its input shape).
5. **Units:** microvolts. Version 27 keeps volts, but it aligns the data
   before the model; PA-TCNet does not align, and its first nonlinearity
   (ELU) acts on the raw scale, where volts would make it linear.

## 5. Notes from setting it up

- The model built with these settings has 71,132 parameters for Liu 2024 and
  74,762 for Jia 2019; inputs of 30 x 1000 and 63 x 1708 give two logits.
- On a Liu 2024 subject the two class templates of PGTC (built from the ROI
  time courses as in `physiology.py`) have a cosine similarity of 0.89, and
  the class thresholds come out at about 0.55. The templates are averaged raw
  time courses, so they differ little between the classes.
- In a trial fold (subject 2, 30 epochs), PGTC gave a target to 88% of the
  held-out subject's trials in the first epoch after the warm-up and 90%
  afterwards: with a confidence threshold of 0.60 most predictions pass.

## Files

```
settings.py      the settings above: subjects, shapes, Table 2, Table 3, training
preparation.py   the trials as the model receives them
channels.py      the ROI groups and channel names, in the structure the code expects
network.py       the model built with the paper's settings
training.py      one leave-one-subject-out fold, ported from train.py; both scores
experiment.py    all folds of a dataset, as train_loso
original/        model.py, pgtc.py, physiology.py from the published code, unchanged
```
