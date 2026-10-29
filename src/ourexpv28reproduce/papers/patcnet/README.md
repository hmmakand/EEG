# PA-TCNet: reproduction

Wang X., Zhao Y., He D., Xia Q., Li G., Wang N., Peng N., Jiang B. (2026).
"PA-TCNet: Pathology-Aware Temporal Calibration with Physiology-Guided Target
Refinement for Cross-Subject Motor Imagery EEG Decoding in Stroke Patients".
arXiv:2604.16554. Code: `src/temp/PA-TCNet-main` (model, PGTC, ROI templates,
training loop; no data loaders, no settings file, no utilities).

Status: **ready to run** (specification, data preparation, model, training
and scoring), including BCI2a and within-subject extensions. No full run has been made yet.

## Default: comparison with version 27

PATCNet defaults to `--cohort version27`: the same subjects, task labels, windows, trial groups
and ten within-subject test folds as version 27, through `datascripts/version27.py`.
PATCNet preprocessing (8–30 Hz, common average reference, baseline removal, microvolts), model and
adaptation training are retained. The input dimensions/classifier are adjusted to the shared samples.

| Dataset | Subjects | Task | Model input |
|---|---|---|---|
| Liu 2024 | 46; exclude 5, 6, 15, 49 | rest (0) vs imagery (1), two 1.9 s samples per usable trial | 29 × 475 |
| BCI2a | all 9 | left (0) vs right (1), 2 s from cue | 22 × 500 |
| Jia 2019 | all 15 | left (0) vs right (1), full epoch | 63 × 1708 |

Liu imagery starts 1 s after its own trial onset and rest starts 4 s after it. Trials without a
complete rest window are skipped, exactly as in version 27. Both samples of a trial stay in the
same fold. The same stratified trial splitting function/seed (42) is used by the other baselines.
Nine folds form the labeled source set and one is the unlabeled PGTC target/test set.

Liu's 29-channel montage excludes the all-zero CPz reference. Its PGTC midline ROI therefore uses
FCz and Cz; the left/right ROIs remain unchanged. No electrode is fabricated. Jia remains
confidence-only because channel names are unavailable. **PGTC uses unlabeled test inputs during
training** in both protocols; report this transductive adaptation when comparing methods.

Use `final_epoch` as the main metric. The test-selected `as_published` score is diagnostic only.
Describe these experiments as **PATCNet evaluated under our common experimental protocol**, rather
than reproduction of the paper's reported numbers.

From the repository root, using an environment with the project dependencies:

```bash
bash src/ourexpv28reproduce/papers/patcnet/run_all.sh
```

This runs all three datasets under `loso` and `within`, saves logs, and draws figures.
The script defaults to `/root/autodl-tmp/Experiment/EEG/` paths. Override `LIU`, `BCI`, `JIA`, `OUT`
to use another location. `--save-models` remains unsupported.

For a single run:

```bash
python -u -m src.ourexpv28reproduce.run --paper patcnet --dataset liu2024 \
  --cohort version27 --protocol within --quiet --data-dir "$LIU" --output-dir "$OUT"
```

Results are named `PATCNet_<dataset>_version27_<LOSO|Within>_seed2.json`, with a `.summary.txt`.
Within-subject JSON includes trial groups, split indices, fold histories and pooled subject scores.
Results using paper inputs have different names, so they are not mixed with comparison outputs.

## Optional original-paper inputs

`--cohort published` explicitly selects the original left/right stroke samples and the paper's
24-patient Liu cohort (including the duplicate 47/49). `--cohort distinct` removes subject 49.
These modes use Liu 30 × 1000 and Jia 63 × 1708 inputs. The paper reports LOSO only.
The remaining paper-specific specifications below describe these optional reproduction modes.

```bash
python -u -m src.ourexpv28reproduce.run --paper patcnet --dataset liu2024 \
  --cohort published --protocol loso --quiet --data-dir "$LIU" --output-dir "$OUT"
```

### The two scores of every run

Every fold trains for all 200 epochs and evaluates the held-out subject after
each epoch. Two scores are computed from the same run:

| Score | How it is chosen | Uses the test labels |
|---|---|---|
| **as published** | The published code stops when the held-out subject's loss has not improved for 30 epochs, and reports the epoch with the highest accuracy on the held-out subject up to then | Yes |
| **final epoch** | The score after epoch 200 | No |

Training is the same either way (stopping only shortens it), so one run gives
both from the same training trajectory (checkpoint utility behavior is assumed; see the audit below). For explicit paper-cohort LOSO runs, the "as published" score is the one to set beside the paper's
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

The paper reports leave one subject out only; no within-subject experiment. Our BCI2a and within-subject runs are extensions. Accuracy, Cohen's
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
command.py       the command-line options, result name and summary table
original/        model.py, pgtc.py, physiology.py from the published code, unchanged
```

## Methodology audit and extensions

Checked against the local paper PDF and `src/temp/PA-TCNet-main/Model/`:

- `original/model.py`, `pgtc.py`, and `physiology.py` match the released files except for attribution comments.
- Model settings match Table 2. Stroke adaptation parameters match Table 3 (rather than the conflicting CLI defaults).
- The source warm-up, Adam settings, source/target loss, target-loader cycling, eval-mode initialization,
  train-mode target refresh after each optimizer step, and persistence of accepted pseudo-labels match `train.py`.
- Fixed evaluation/calibration loaders to retain singleton final batches. The original only drops those batches
  for training; the previous port also dropped them in evaluation/calibration.
- The original reads target labels for checkpoint selection and early stopping despite the paper's description
  of label-free target adaptation. `as_published` preserves this test-selected diagnostic; use `final_epoch`
  as the main score selected without test labels. Both use **unlabeled test inputs** for transductive PGTC.
- The release omits loaders, settings, `BestModelSaver`, and `EarlyStopping`. Early-stop behavior and
  checkpoint tie handling are conventional assumptions, so exact numerical reproduction is not guaranteed.
- ICA remains omitted because component selection is unspecified. Jia remains confidence-only because electrode
  names are unavailable. The paper's pooling, band (8–30 Hz), CAR and baseline removal are retained.

The CLI accepts `--dataset {liu2024,bci2a,jia2019}`, `--protocol {loso,within}` (default LOSO),
and `--cohort {version27,published,distinct}` (default version27).

BCI2a uses version 27's nine subjects (all of them), binary left/right labels, 22 EEG channels and
2 seconds after the cue (500 samples). All nine motor ROI electrodes are present. Adaptation uses the released
code's defaults: alpha 0.95, confidence 0.90, ROI floor 0.70, warm-up 10; the paper has no BCI2a settings.
The MATLAB BCI2a reader returns microvolts already: the PATCNet preparation leaves that scale unchanged,
whereas Liu/Jia are converted from volts to microvolts. This handling is local to PATCNet.

Within subject uses version 27's grouped stratified trial folds in the default comparison mode.
The optional paper modes have one sample per original trial and use ten stratified trial folds.
Predictions are pooled over test folds per subject, then dataset metrics average subjects.

Validation: the original-paper tests remain, with additional checks for shared cohorts, Liu input shape,
exact grouped fold indices, and real Liu comparison sample labels/groups and short PGTC training.
Full 200-epoch experiments have not been launched.
