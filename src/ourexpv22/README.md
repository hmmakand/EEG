# Version twenty-two: left/right motor imagery from EEG graphs

Version twenty-two is a standalone copy of version twenty-one with the same
pipeline and default model. It adds a multi-seed baseline (`--seed`) and two
subject-invariance training losses (`--invariance`), neither of which improved
the result; see [READMEFORMETHOD.md](READMEFORMETHOD.md). It classifies
left-hand versus right-hand motor imagery from BCI Competition IV 2a. Each
trial becomes a graph of 22 electrodes that also carry their time signals, and
a custom graph transformer with a temporal encoder
([model/custom/](model/custom/)) classifies it.

The main experiment is leave-one-subject-out (LOSO): train on seven subjects,
test on the eighth. The current result is **0.791 mean accuracy over five
seeds** (0.785 to 0.802). The per-subject table under [Results](#results) is
the seed-12345 run (0.801).

This README follows the pipeline in order:

1. [Set up](#1-set-up)
2. [Prepare the original signals and the folds](#2-prepare-the-original-signals-and-the-folds)
3. [Generate the synthetic signals](#3-generate-the-synthetic-signals)
4. [How a trial becomes a graph](#4-how-a-trial-becomes-a-graph)
5. [The model](#5-the-model)
6. [Train and evaluate (LOSO)](#6-train-and-evaluate-loso)
7. [Results](#results) and [Ablation](#ablation-of-the-three-paths)
8. [Other trainers](#other-trainers)
9. [Reporting notes](#reporting-notes)
10. [Tests](#tests)
11. [History](#history)

The folder is standalone: it imports no other experiment, and its Python module
name is `src.ourexpv22`.

## Folder map

| Folder | Purpose |
| --- | --- |
| [datasetsynthetic/](datasetsynthetic/) | Step 2 and 3: extract trials, save fold plans, train the diffusion generators, write synthetic signals; `evaluate.py` checks the synthetic signals against the originals |
| [dataset/](dataset/) | Step 4: turn one trial signal into one graph |
| [datacustom/](datacustom/) | Read a saved signal bundle and build the graphs of one fold |
| [model/](model/) | Step 5: the custom model (`custom/`) and the earlier GATv2 (`gat.py`) |
| [train/](train/) | Step 6: trainers, settings, metrics, result files |
| [tests/](tests/) | Unit and regression tests |

Signals and results live outside the repository:

| Path | Content |
| --- | --- |
| `/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a` | Raw MATLAB files `A01T.mat` ... `A09T.mat` |
| `/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv22/datasyn/fold_cross` | Prepared signal bundle used for LOSO (original and synthetic signals, fold manifest) |
| `/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv22/outputs` | LOSO result JSON and log files |

The commands below use two shell variables:

```bash
RAW=/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a
DATA=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv22
```

## 1. Set up

Use Python 3.10 or newer. From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv22/requirements.txt
python -m pip install -r src/ourexpv22/datasetsynthetic/eegdiffuser/requirements.txt  # only for step 3
```

Subjects 1, 2, 3, 5, 6, 7, 8 and 9 are used; subject 4 is excluded. Only the
training recordings (`A0xT.mat`) are read.

## 2. Prepare the original signals and the folds

Preparation extracts every left/right trial and saves the fold plans. It builds
no graphs and trains nothing.

```bash
python -m src.ourexpv22.datasetsynthetic.prepare \
  --data-dir "$RAW" \
  --output "$DATA/datasyn/fold_cross"
```

| Setting | Value |
| --- | --- |
| Sessions | Indices 3 to 8 (zero-based), the six motor-imagery runs |
| Channels | Indices 0 to 21, the 22 EEG electrodes |
| Classes | Label 1 becomes 0 (left hand), label 2 becomes 1 (right hand); other labels are skipped |
| Trial signal | 4 seconds from the trial start at 250 Hz: 1000 samples by 22 channels |
| Trials per subject | 144 (72 per class) |
| Cross-subject plan | Leave one subject out: eight folds |
| Within-subject plan | Ten shuffled folds per subject, seed 42 |
| Synthetic trials planned | 50 per class per generator |

The trial start is 2 seconds before the cue, so the saved signal covers 2
seconds of rest and the first 2 seconds of imagery.

Preparation writes `original/` (signals and trial identities), `inputs/` (the
originals each generator may use), `protocols/*.plan.csv`, `jobs.json` and
`metadata.json`. Rerunning it on an identical bundle verifies and reuses it.
Details: [datasetsynthetic/README.md](datasetsynthetic/README.md).

## 3. Generate the synthetic signals

One diffusion model (EEGDiffuser) is trained per subject and generates 100
synthetic trials for that subject. The LOSO trainer does not train on the
synthetic trials by default (see step 6), but it reads the fold manifest that
this step publishes, so the step must be complete.

```bash
python -u -m src.ourexpv22.datasetsynthetic.generate \
  --bundle "$DATA/datasyn/fold_cross" \
  --protocol cross_subject \
  --epochs 100 \
  --device cuda
```

| Setting | Value |
| --- | --- |
| Generators | Eight, one per subject |
| Generator training data | That subject's 144 originals: 134 for training, 10 for checkpoint selection |
| Epochs | 100 |
| Batch size | 1 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.05, cosine schedule to 1e-6 |
| Model | Embedding 512, patch size 5, four transformer blocks, 16 heads, four global tokens, no self-conditioning |
| Diffusion | 1000 steps, linear noise schedule, classifier-free guidance scale 4.0 |
| Output | 50 trials per class per subject, same shape as the originals |
| Seed | 8888 |

When generation finishes it publishes `protocols/cross_subject.csv`, the
manifest the trainer reads. Add `--resume` to continue an interrupted run with
the same settings.

The bundle currently at `$DATA/datasyn/fold_cross` was produced with these
settings by version sixteen's copy of the same generator and is reused here.
Its synthetic trials do not carry the real class signal (see
[Diagnosis of the synthetic trials](#diagnosis-of-the-synthetic-trials)).

A second bundle, `$DATA/datasynv1/fold_cross`, is prepared for retrained
generators: 1000 epochs, guidance 1.5, generation from the final epoch. It
holds the same original trials and folds. See
[READMEFORMETHOD.md](READMEFORMETHOD.md) for the pilot behind these settings:

```bash
python -u -m src.ourexpv22.datasetsynthetic.generate \
  --bundle "$DATA/datasynv1/fold_cross" --protocol cross_subject \
  --epochs 1000 --cfg-scale 1.5 --sample-batch-size 10 \
  --checkpoint-selection final --checkpoint-interval 50 --device cuda
``` Its
within-subject generators were not run, so only the LOSO manifest exists in it.
Details: [datasetsynthetic/eegdiffuser/README.md](datasetsynthetic/eegdiffuser/README.md).

### What one LOSO fold contains

| Part | Trials | Used by default |
| --- | ---: | --- |
| Training, original (seven subjects) | 1008 | Yes |
| Training, synthetic (the same seven subjects) | 700 | No; add `--synthetic` in step 6 |
| Testing, original (held-out subject) | 144 | Yes |

The held-out subject's originals and synthetic trials never enter training.

## 4. How a trial becomes a graph

Graphs are built in memory when a fold is loaded; no graph files are written.
Original and synthetic trials go through the same steps. The rules are in
[dataset/config.py](dataset/config.py) (`default_dataset_config()`).

| Step | Rule |
| --- | --- |
| Align | Each subject's trials are multiplied by the inverse square root of that subject's mean covariance (Euclidean alignment). No labels are used. Originals and each generator's output are aligned separately |
| Filter | Fifth-order Butterworth band-pass, zero-phase, 8 to 30 Hz |
| Select window | Seconds 2 to 4 of the filtered trial (500 samples), the part after the cue |
| Node features | Six band powers per electrode: 8-12, 12-16, 16-20, 20-24, 24-28, 28-32 Hz (Welch, 2-second Hann window, Simpson integration) |
| Edges | Every ordered electrode pair, no self-loops: 462 edges |
| Edge features | Phase-locking value (PLV) and 8-30 Hz covariance of the pair |
| Node signal | The aligned, filtered, windowed signal of each electrode (500 samples), for the temporal encoder |

Each graph holds `x` `[22, 6]`, `edge_index` `[2, 462]`, `edge_attr` `[462, 2]`,
`edge_weight` `[462]` (the PLV alone), `signal` `[22, 500]`, the label `y`, and
subject, session and trial identifiers.

PLV is `abs(mean(exp(1j * phase_difference)))` over the window, with phases
from the Hilbert transform. The 8-30 Hz filter is applied before the band
filters, so the band powers describe the filtered signal.

## 5. The model

The default classifier is the custom GPS + GRIT hybrid in
[model/custom/](model/custom/), with settings in
[model/custom/config.py](model/custom/config.py). `train/setup.py` builds it.
Its logits are the sum of three paths: the graph head, the wide path and the
temporal encoder.

| Part | Setting |
| --- | --- |
| Input | Log of the band powers; edge features used as given |
| Node encoder | Linear layer plus a learned embedding per electrode |
| Pair stream | One vector per ordered electrode pair: encoded edge features plus random-walk encodings (8 steps, weighted by the edge features) |
| Layers | Two. Each runs a GatedGCN branch over the edges and GRIT attention over all pairs, then sums them |
| Sizes | Hidden size 32, four attention heads |
| Regularisation | Dropout 0.3, attention dropout 0.2, BatchNorm |
| Head | Mean pooling of nodes and of pairs, then a two-layer MLP |
| Wide path | A linear layer on the flattened node and edge inputs, added to the logits (dropout 0.5). It lets the model weight a specific electrode or electrode pair |
| Temporal encoder | Reads the node signals. 8 temporal filters of 25 samples (0.1 s) shared by all electrodes, then 16 learned spatial filters across the electrodes, squaring, average pooling (75 samples, stride 15), log, dropout 0.7, and a linear layer added to the logits |
| Parameters | 44,822 |

The temporal encoder is the ShallowConvNet idea: a spatial filter followed by
log power is what CSP computes, here with the temporal and spatial filters
learned together with the graph model. `temporal` in the model configuration
selects the path: `"head"` (default, described above), `"node"` (per-electrode
log power added to the node embeddings, no spatial filters), `"both"`,
`"edge"` (learned-band covariance of every electrode pair added to the pair
stream of the attention), `"edge+head"`, or `"none"`.

Cropped training is available but off: `temporal_crop` in the model
configuration (`TEMPORAL_CROP` in [train/setup.py](train/setup.py)) trains the
temporal encoder on a random window per trial and averages over windows at
evaluation. It did not improve validation accuracy; see [History](#history).

The electrode embedding, the wide path and the temporal encoder assume that
every graph lists the same 22 electrodes in the same order.

`--model gat` selects the earlier GATv2 model ([model/gat.py](model/gat.py)):
three GATv2 layers with 22 hidden channels and three heads, GraphNorm, mean
pooling, dropout 0.5. It uses the edges but not the edge features or the signals.

## 6. Train and evaluate (LOSO)

```bash
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
NAME=Version22_GritGPS_LOSO_OriginalOnly_20epochs_Batch32_Ens3_${run_stamp}
python -u -m src.ourexpv22.train.train_syn_loso \
  --bundle "$DATA/datasyn/fold_cross" \
  --device cuda \
  --output "$DATA/outputs/$NAME.json" 2>&1 | tee "$DATA/outputs/$NAME.log"
```

All eight folds take about 8 minutes on a GTX 1060.

| Setting | Value | Option |
| --- | --- | --- |
| Model | Custom | `--model custom` or `gat` |
| Training data | Original trials of the seven training subjects (1008) | `--synthetic` adds the 700 synthetic trials |
| Model paths | Graph head, wide path and temporal encoder | `--paths graph wide temporal` (any subset) |
| Epochs | 20 | `--epochs` |
| Batch size | 32, training batches shuffled | `--batch-size` |
| Optimizer | Adam, learning rate 0.001, weight decay 0.001 | |
| Loss | Cross-entropy | |
| Models per fold | 3, seeds 12345, 12346, 12347; class probabilities averaged | `--ensemble` |
| Logit centring | On: the held-out subject's mean logit is subtracted before the decision | `--no-center-logits` |
| Evaluation | Once, after the final epoch, on the held-out subject's 144 originals | |
| Subjects | All eight | `--subjects 1 2 3` |
| Seed | 12345 (the three models use 12345, 12346, 12347) | `--seed` |
| Invariance loss | None | `--invariance adversarial` or `contrastive`, `--invariance-weight` |

There is no validation set during training and no early stopping; the schedule
is fixed in advance. Training settings are in
[train/config.py](train/config.py) and [train/config_syn.py](train/config_syn.py);
the LOSO defaults above are set in
[train/train_syn_loso.py](train/train_syn_loso.py).

The result JSON holds, per held-out subject, accuracy, balanced accuracy, macro
F1 and the confusion matrix, plus the settings, the manifest hash and the
package versions. No model weights are saved. Without `--output`, the file goes
to `outputs/` inside this folder with a timestamped name.

## Results

One run with the settings above:
`Version21_Ablation_GraphWideTemporal_LOSO_OriginalOnly_20epochs_Batch32_Ens3_20261004T051101Z.json`
in `$DATA/outputs/`. It reproduces the version-twenty run exactly (same data,
settings and seeds), so it is a repeat, not an independent second sample.

| Held-out subject | Version 19 | Versions 20 and 21 | CSP + LDA with alignment |
| --- | ---: | ---: | ---: |
| S1 | 0.681 | 0.833 | 0.78 |
| S2 | 0.639 | 0.715 | 0.65 |
| S3 | 0.896 | 0.917 | 0.90 |
| S5 | 0.625 | 0.681 | 0.67 |
| S6 | 0.681 | 0.778 | 0.67 |
| S7 | 0.542 | 0.667 | 0.70 |
| S8 | 0.868 | 0.896 | 0.92 |
| S9 | 0.889 | 0.924 | 0.88 |
| **Mean** | **0.727** | **0.801** | **0.77** |

923 of 1152 held-out trials are correct (macro F1 0.801). Sampling noise on
1152 trials is about ±0.012. Predictions are balanced between the classes
(573 and 579).

The CSP + LDA column is a separate classical analysis on the same 2-4 second
window and the same original training trials. The model is 0.031 above it
on the mean and above it on six of the eight subjects (subjects 7 and 8 are
below). This is one run of each; the difference is about 36 trials.

## Ablation of the three paths

The logits are the sum of the graph head, the wide path and the temporal
encoder. Each row is a full LOSO run with only the named paths enabled
(`--paths`), everything else as in step 6. Files:
`Version21_Ablation_*_20261004T051101Z.json` in `$DATA/outputs/`.

| Paths | S1 | S2 | S3 | S5 | S6 | S7 | S8 | S9 | Mean | Correct of 1152 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph head only | 0.653 | 0.688 | 0.833 | 0.653 | 0.639 | 0.569 | 0.840 | 0.896 | **0.721** | 831 |
| Wide path only | 0.750 | 0.639 | 0.917 | 0.688 | 0.646 | 0.590 | 0.889 | 0.819 | **0.742** | 855 |
| Temporal encoder only | 0.819 | 0.701 | 0.896 | 0.632 | 0.743 | 0.694 | 0.896 | 0.896 | **0.785** | 904 |
| Graph + wide | 0.729 | 0.667 | 0.882 | 0.660 | 0.660 | 0.632 | 0.882 | 0.889 | **0.750** | 864 |
| Graph + temporal | 0.806 | 0.708 | 0.889 | 0.660 | 0.778 | 0.694 | 0.910 | 0.903 | **0.793** | 914 |
| Wide + temporal | 0.819 | 0.674 | 0.903 | 0.688 | 0.729 | 0.681 | 0.896 | 0.889 | **0.785** | 904 |
| All three (default) | 0.833 | 0.715 | 0.917 | 0.681 | 0.778 | 0.667 | 0.896 | 0.924 | **0.801** | 923 |

- The temporal encoder carries most of the result: alone it reaches 0.785,
  against 0.721 for the graph head and 0.742 for the wide path.
- Adding the graph head to the temporal encoder gives 0.793, and all three
  give 0.801. Those steps are 10 and 9 trials out of 1152, each within the
  sampling noise of about ±0.012, from single runs with the same seeds. The
  ordering is consistent, but this does not establish that the graph paths
  add to the temporal encoder.
- Without the temporal encoder, the graph head and wide path together reach
  0.750, below the CSP + LDA reference (0.77).

To run one ablation:

```bash
python -u -m src.ourexpv22.train.train_syn_loso \
  --bundle "$DATA/datasyn/fold_cross" --device cuda \
  --paths graph temporal \
  --output "$DATA/outputs/my_ablation.json"
```

## Diagnosis of the synthetic trials

Why the synthetic trials do not help, measured on the bundle with CSP + LDA
and logistic regression (means over the eight subjects). The tool is
[datasetsynthetic/evaluate.py](datasetsynthetic/evaluate.py):

```bash
python -m src.ourexpv22.datasetsynthetic.evaluate \
  --bundle "$DATA/datasyn/fold_cross" \
  --output "$DATA/outputs/Version21_SyntheticEvaluation_fold_cross.json"
```

It reads saved signals only and takes about a minute. Run it on any new
bundle before training a classifier on its synthetic trials.

| Check | Result | Reading |
| --- | --- | --- |
| Train on a subject's real trials, test on real (seconds 2-4) | 0.77 | The real class signal |
| Train on real, test on that subject's synthetic trials | 0.59 | The synthetic trials carry little of the real class signal |
| Train on synthetic, test on real | 0.57 | Same, in the other direction |
| Train on synthetic, test on synthetic | 0.70 | The synthetic classes differ from each other, in their own way |
| Same, on seconds 0-2 (before the cue) | 0.72 synthetic, 0.51 real | The synthetic class difference is present before the cue, where real trials have none |
| Tell real from synthetic using five band powers | 0.99 | Synthetic trials are trivially recognisable |

The spectrum explains the last row: synthetic trials have about 2.7 times the
real power at 13-30 Hz, 4 times at 30-45 Hz and 0.4 times at 1-4 Hz, and
their trial-to-trial variation in band power is 0.3 to 0.8 times that of real
trials, depending on the band.

So the generator has learned a class-dependent pattern that is not the
post-cue effect of real trials, with the wrong spectrum and too little variety.
A classifier that reads the signals learns that pattern, which is why training
accuracy reaches 1.00 and held-out accuracy falls. The generators were trained
for 100 epochs (the reference default is 1000) and were still improving: six of
the eight had their lowest validation loss at epoch 99. They were sampled
from the ordinary weights, not the EMA weights, with classifier-free guidance
scale 4.0.

### Re-sampling the existing generators does not fix it

[datasetsynthetic/resample.py](datasetsynthetic/resample.py) draws new trials
from the trained generators with other sampling settings, without retraining
and without changing the bundle. Subjects 1, 3 and 8, 30 trials per class, 250
sampling steps; results in `$DATA/resampling_20261004T075522Z/`.

| Guidance scale, weights | real->syn | syn->real | syn->syn before the cue | real vs syn | 13-30 Hz power, syn / real |
| --- | ---: | ---: | ---: | ---: | ---: |
| Real trials, for reference (real->real 0.893) | | | 0.538 | | 1 |
| 4.0, ordinary (the bundle, 1000 steps) | 0.607 | 0.630 | 0.757 | 0.989 | 2.19 |
| 4.0, ordinary | 0.644 | 0.551 | 0.694 | 0.982 | 2.14 |
| 2.5, ordinary | 0.583 | 0.546 | 0.572 | 0.984 | 2.13 |
| 1.5, ordinary | 0.567 | 0.523 | 0.550 | 0.984 | 2.12 |
| 1.0, ordinary | 0.561 | 0.500 | 0.517 | 0.984 | 2.11 |
| 1.5, EMA | 0.494 | 0.523 | 0.544 | 1.000 | about 86,000 |

- Lower guidance removes the class difference before the cue, and with it
  the little class signal the synthetic trials had. The guidance was
  producing most of the class difference, not the generator's knowledge of
  the real one.
- The spectrum and the real-versus-synthetic score do not depend on the
  sampling settings, so they come from the trained weights.
- The EMA weights are unusable after 100 epochs: their output has tens of
  thousands of times the real power.

No sampling setting passes, so better synthetic trials need retrained
generators.

## Other trainers

Two more trainers share the dataset rules, the model and the optimizer
settings. Neither has been run with the current defaults.

**Within-subject KFold on original trials** (`train/train.py`). It reads the raw
MATLAB files directly and needs no bundle.

```bash
python -u -m src.ourexpv22.train.train --data-dir "$RAW" --device cuda
```

Defaults: ten shuffled folds per subject (seed 42), 249 epochs, batch 32, one
model per fold (seed 12345). The epoch with the highest test-fold accuracy is
reported, so these scores are not independent test estimates. Options:
`--subjects`, `--folds`, `--epochs`, `--model`, `--output`.

**Saved within-subject folds with synthetic trials** (`train/train_syn.py`). It
needs a bundle whose within-subject generators have been run (step 3 with
`--protocol within_subject`). Defaults: 100 epochs, batch 8, best test epoch
reported. See [datacustom/README.md](datacustom/README.md).

**Earlier baseline.** The GATv2 results of earlier versions used the legacy
dataset rules (`legacy_dataset_config()`: whole 4-second trial, eight bands,
PLV graph thresholded at 0.35, no alignment) with `model="gat"`,
`train_shuffle=False` and `weight_decay=0`.

## Reporting notes

- **Unlabeled test-subject data is used.** Alignment and logit centring both
  use the held-out subject's own trials without their labels. State this when
  reporting; the result JSON records it under `unlabeled_test_subject_use`.
- **How settings were chosen.** Model settings and the LOSO schedule were
  chosen by holding out one training subject at a time; the LOSO test subjects
  were not used. The 2-4 second window and the feature types were chosen from
  classical analyses over all eight subjects, so the LOSO result is mildly
  optimistic in that respect.
- **One run.** The result above is a single run. The three models per fold
  share the same data and differ only in seed.
- **Synthetic trials are not used by default.** With the temporal encoder the
  model fits the synthetic trials perfectly and validation accuracy drops by
  about five points, so the LOSO trainer leaves them out.

## Tests

```bash
python -m unittest discover -s src/ourexpv22/tests
```

46 tests cover the dataset rules, alignment, the bundle loaders, the model
paths, the trainers, the synthetic-signal checks and the regression fixtures
of the legacy pipeline. They need no GPU and no
real data.

## History

| Version | LOSO accuracy | What changed |
| --- | ---: | --- |
| 10 to 17 | 0.499 to 0.512 | GATv2 on thresholded PLV graphs of the whole trial. At chance |
| 18, first run | 0.498 | Custom model on the same graphs. At chance |
| 18 | 0.628 | Seconds 2-4 only, log and per-trial standardisation of band powers, PLV on every pair as edge weight, six bands, shuffled batches, weight decay |
| 19 | 0.727 | Per-subject alignment, covariance edge features, electrode-aware wide path, logit centring, three models per fold |
| 20 | 0.801 | Temporal encoder on the electrode signals (learned temporal and spatial filters, log power), originals-only training |
| 21 | 0.801 | Same model and data. Adds the path ablation and optional cropped training |

Why versions 10 to 17 were at chance:

| Problem | Evidence |
| --- | --- |
| Half of each trial is before the cue | CSP + LDA scores 0.52 within subject on seconds 0-2 and 0.79 on seconds 2-4 |
| Raw band powers differ up to 15 times between subjects | The custom model could not fit even the training subjects |
| The graph kept only which PLV values pass 0.35 | A linear classifier on band power plus PLV values scores 0.69 on LOSO |
| Two of the eight bands were empty | The 8-30 Hz filter leaves about zero power above 32 Hz |
| Training batches were not shuffled | The saved LOSO order groups trials by subject |

What version nineteen added, measured on four held-out training subjects (576
trials, noise about 0.02):

| Change | Validation accuracy |
| --- | --- |
| Version-eighteen rules | 0.65 |
| Per-subject alignment | about 0.69 |
| Covariance next to PLV on the edges | about +0.01, within noise |
| Wide path in the head | 0.70 to 0.73 |
| Plain log inputs with logit centring | 0.72 to 0.75 |

What version twenty added, measured the same way (centred accuracy at epochs
10, 20 and 30):

| Model | Training data | Validation accuracy |
| --- | --- | --- |
| Version-nineteen model | Originals and synthetic | 0.75, 0.73, 0.73 |
| Version-nineteen model | Originals only | 0.76, 0.75, 0.72 |
| Temporal `node` path | Originals and synthetic | 0.75, 0.73, 0.74 |
| Temporal `node` path | Originals only | 0.75, 0.79, 0.76 |
| Temporal `head` path | Originals and synthetic | 0.75, 0.72, 0.70 |
| Temporal `head` path | Originals only | 0.76, 0.77, 0.77 |
| Temporal `head` path, dropout 0.7 (default) | Originals only | 0.78, 0.78, 0.78 |

Also tried with the `head` path and not kept: 16 temporal filters, a 51-sample
kernel, longer pooling, no wide path, and the `both` mode. They scored the same
or lower, or were less stable across epochs.

Cropped training, tried in version twenty-one on the same four held-out training
subjects (centred accuracy at epochs 10, 20 and 30, originals only):

| Temporal encoder input | Validation accuracy |
| --- | --- |
| Whole 2-second window (default) | 0.77, 0.78, 0.78 |
| Random 1.75-second crops (437 samples) | 0.77, 0.78, 0.77 |
| Random 1.5-second crops (375 samples) | 0.78, 0.78, 0.75 |
| Random 1-second crops (250 samples) | 0.79, 0.78, 0.78 |

All four are within the noise of about 0.02, so cropping stays off.

Temporal features inside the graph attention, tried in version twenty-one the
same way. The `edge` mode adds the covariance of every electrode pair in each
learned band to the pair stream that conditions the attention, and each
electrode's log power to its node embedding:

| Temporal mode | Validation accuracy |
| --- | --- |
| `head` (default): spatial filters outside the graph | 0.77, 0.78, 0.78 |
| `edge`: learned-band covariance into the attention | 0.75, 0.74, 0.73 |
| `edge` without the wide path | 0.74, 0.75, 0.71 |
| `edge` with 16 temporal filters | 0.73, 0.77, 0.71 |
| `edge+head` | 0.77, 0.77, 0.76 |

Routed through the attention, the temporal features score about the same as
the graph model without them and less steadily; added to the `head` path they
bring nothing. The mode stays in the code and off.

Electrode-aware readout, tried in version twenty-one the same way. With
`readout="electrode"` the graph head replaces mean pooling by four learned,
signed weightings over the 22 electrodes and the 484 electrode pairs, so it can
weight or contrast specific electrodes:

| Model (originals only) | Mean pooling | Electrode-aware readout |
| --- | --- | --- |
| Graph head only | 0.72, 0.72, 0.71 | 0.72, 0.69, 0.71 |
| Graph head with temporal `edge` features | 0.74, 0.75, 0.71 | 0.71, 0.69, 0.72 |
| Graph head, wide path and temporal `edge` features | 0.75, 0.74, 0.73 | 0.72, 0.70, 0.69 |
| Default model (graph, wide, temporal `head`) | 0.77, 0.78, 0.78 | 0.75, 0.72, 0.71 |

The readout raises training accuracy from about 0.80 to 0.99 and lowers or
leaves unchanged the accuracy on held-out subjects. The graph head is
therefore not held back by its readout being too weak; giving it more freedom
makes it fit the training subjects instead of transferring. The option stays
in the code and off.

Tried in version nineteen and not kept, because they scored the same or lower: separate mu and
beta covariance bands (`connectivity.covariance_bands`), band powers per time
segment (`features.segments`), one or three layers, stronger dropout. The
options remain in the configuration.

The files in this folder's `output/` are GATv2 results inherited from version
sixteen, not version-twenty-one runs.
