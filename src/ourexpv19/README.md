# Version nineteen: left/right motor imagery from EEG graphs

Version nineteen classifies left-hand versus right-hand motor imagery from
BCI Competition IV 2a. Each trial becomes a graph of 22 electrodes, and a
custom graph transformer ([model/custom/](model/custom/)) classifies it.

The main experiment is leave-one-subject-out (LOSO): train on seven subjects
plus synthetic trials, test on the eighth. The current result is **0.727 mean
accuracy** (838 of 1152 held-out trials). See [Results](#results).

This README follows the pipeline in order:

1. [Set up](#1-set-up)
2. [Prepare the original signals and the folds](#2-prepare-the-original-signals-and-the-folds)
3. [Generate the synthetic signals](#3-generate-the-synthetic-signals)
4. [How a trial becomes a graph](#4-how-a-trial-becomes-a-graph)
5. [The model](#5-the-model)
6. [Train and evaluate (LOSO)](#6-train-and-evaluate-loso)
7. [Results](#results)
8. [Other trainers](#other-trainers)
9. [Reporting notes](#reporting-notes)
10. [Tests](#tests)
11. [History](#history)

The folder is standalone: it imports no other experiment, and its Python module
name is `src.ourexpv19`.

## Folder map

| Folder | Purpose |
| --- | --- |
| [datasetsynthetic/](datasetsynthetic/) | Step 2 and 3: extract trials, save fold plans, train the diffusion generators, write synthetic signals |
| [dataset/](dataset/) | Step 4: turn one trial signal into one graph |
| [datacustom/](datacustom/) | Read a saved signal bundle and build the graphs of one fold |
| [model/](model/) | Step 5: the custom model (`custom/`) and the earlier GATv2 (`gat.py`) |
| [train/](train/) | Step 6: trainers, settings, metrics, result files |
| [tests/](tests/) | Unit and regression tests |

Signals and results live outside the repository:

| Path | Content |
| --- | --- |
| `/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a` | Raw MATLAB files `A01T.mat` ... `A09T.mat` |
| `/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv19/datasyn/fold_cross` | Prepared signal bundle used for LOSO (original and synthetic signals, fold manifest) |
| `/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv19/outputs` | LOSO result JSON and log files |

The commands below use two shell variables:

```bash
RAW=/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a
DATA=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv19
```

## 1. Set up

Use Python 3.10 or newer. From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv19/requirements.txt
python -m pip install -r src/ourexpv19/datasetsynthetic/eegdiffuser/requirements.txt  # only for step 3
```

Subjects 1, 2, 3, 5, 6, 7, 8 and 9 are used; subject 4 is excluded. Only the
training recordings (`A0xT.mat`) are read.

## 2. Prepare the original signals and the folds

Preparation extracts every left/right trial and saves the fold plans. It builds
no graphs and trains nothing.

```bash
python -m src.ourexpv19.datasetsynthetic.prepare \
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
synthetic trials for that subject.

```bash
python -u -m src.ourexpv19.datasetsynthetic.generate \
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
settings by version sixteen's copy of the same generator and is reused here. Its
within-subject generators were not run, so only the LOSO manifest exists in it.
Details: [datasetsynthetic/eegdiffuser/README.md](datasetsynthetic/eegdiffuser/README.md).

### What one LOSO fold contains

| Part | Trials |
| --- | ---: |
| Training, original (seven subjects) | 1008 |
| Training, synthetic (the same seven subjects) | 700 |
| Testing, original (held-out subject) | 144 |

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

Each graph holds `x` `[22, 6]`, `edge_index` `[2, 462]`, `edge_attr` `[462, 2]`,
`edge_weight` `[462]` (the PLV alone), the label `y`, and subject, session and
trial identifiers.

PLV is `abs(mean(exp(1j * phase_difference)))` over the window, with phases
from the Hilbert transform. The 8-30 Hz filter is applied before the band
filters, so the band powers describe the filtered signal.

## 5. The model

The default classifier is the custom GPS + GRIT hybrid in
[model/custom/](model/custom/), with settings in
[model/custom/config.py](model/custom/config.py). `train/setup.py` builds it.

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
| Parameters | 40,828 |

The electrode embedding and the wide path assume that every graph lists the
same 22 electrodes in the same order.

`--model gat` selects the earlier GATv2 model ([model/gat.py](model/gat.py)):
three GATv2 layers with 22 hidden channels and three heads, GraphNorm, mean
pooling, dropout 0.5. It uses the edges but not the edge features.

## 6. Train and evaluate (LOSO)

```bash
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
NAME=Version19_GritGPS_LOSO_Synthetic_20epochs_Batch32_Ens3_${run_stamp}
python -u -m src.ourexpv19.train.train_syn_loso \
  --bundle "$DATA/datasyn/fold_cross" \
  --device cuda \
  --output "$DATA/outputs/$NAME.json" 2>&1 | tee "$DATA/outputs/$NAME.log"
```

All eight folds take about 10 minutes on a GTX 1060.

| Setting | Value | Option |
| --- | --- | --- |
| Model | Custom | `--model custom` or `gat` |
| Epochs | 20 | `--epochs` |
| Batch size | 32, training batches shuffled | `--batch-size` |
| Optimizer | Adam, learning rate 0.001, weight decay 0.001 | |
| Loss | Cross-entropy | |
| Models per fold | 3, seeds 12345, 12346, 12347; class probabilities averaged | `--ensemble` |
| Logit centring | On: the held-out subject's mean logit is subtracted before the decision | `--no-center-logits` |
| Evaluation | Once, after the final epoch, on the held-out subject's 144 originals | |
| Subjects | All eight | `--subjects 1 2 3` |

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
`Version19_GritGPS_LOSO_Synthetic_20epochs_Batch32_Ens3_20261003T124706Z.json`
in `$DATA/outputs/`.

| Held-out subject | Version 18 | Version 19 | CSP + LDA with alignment |
| --- | ---: | ---: | ---: |
| S1 | 0.653 | 0.681 | 0.78 |
| S2 | 0.563 | 0.639 | 0.65 |
| S3 | 0.583 | 0.896 | 0.90 |
| S5 | 0.500 | 0.625 | 0.67 |
| S6 | 0.632 | 0.681 | 0.67 |
| S7 | 0.507 | 0.542 | 0.70 |
| S8 | 0.785 | 0.868 | 0.92 |
| S9 | 0.806 | 0.889 | 0.88 |
| **Mean** | **0.628** | **0.727** | **0.77** |

838 of 1152 held-out trials are correct (macro F1 0.727). Sampling noise on
1152 trials is about ±0.013. Predictions are balanced between the classes
(606 and 546).

The CSP + LDA column is a separate classical analysis on the same 2-4 second
window using original trials only. Version nineteen does not reach it; the gap
is mainly on subjects 1 and 7.

## Other trainers

Two more trainers share the dataset rules, the model and the optimizer
settings. Neither has been run with the version-nineteen defaults.

**Within-subject KFold on original trials** (`train/train.py`). It reads the raw
MATLAB files directly and needs no bundle.

```bash
python -u -m src.ourexpv19.train.train --data-dir "$RAW" --device cuda
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
- **Synthetic trials.** In validation, training with and without the synthetic
  trials gave the same accuracy within noise.

## Tests

```bash
python -m unittest discover -s src/ourexpv19/tests
```

41 tests cover the dataset rules, alignment, the bundle loaders, the trainers
and the regression fixtures of the legacy pipeline. They need no GPU and no
real data.

## History

| Version | LOSO accuracy | What changed |
| --- | ---: | --- |
| 10 to 17 | 0.499 to 0.512 | GATv2 on thresholded PLV graphs of the whole trial. At chance |
| 18, first run | 0.498 | Custom model on the same graphs. At chance |
| 18 | 0.628 | Seconds 2-4 only, log and per-trial standardisation of band powers, PLV on every pair as edge weight, six bands, shuffled batches, weight decay |
| 19 | 0.727 | Per-subject alignment, covariance edge features, electrode-aware wide path, logit centring, three models per fold |

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

Tried and not kept, because they scored the same or lower: separate mu and
beta covariance bands (`connectivity.covariance_bands`), band powers per time
segment (`features.segments`), one or three layers, stronger dropout. The
options remain in the configuration.

The files in this folder's `output/` are GATv2 results inherited from version
sixteen, not version-nineteen runs.
