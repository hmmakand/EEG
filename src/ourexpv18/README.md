# Version eighteen: standalone supervised EEG classification

Version eighteen is copied from version sixteen. It changes the default
classifier to the custom GPS + GRIT hybrid in [model/custom/](model/custom/)
and changes the default graphs so that the classifier can transfer across
subjects. See [What version eighteen changes](#what-version-eighteen-changes).

This folder contains the dataset pipeline, the custom model, the earlier GATv2
model, and standard supervised training for left-hand versus right-hand motor imagery from BCI Competition IV 2a.
Copy the whole folder anywhere; the code does not import another experiment.
Provide raw MATLAB files separately when using the original-data trainer.
Training uses labeled trial graphs, Adam, and cross-entropy loss.

For **saved within-subject folds with fold-specific synthetic signals**, use
`train/train_syn.py` as described in [datacustom/README.md](datacustom/README.md).
It builds graphs and loaders from the finalized manifest, trains a fresh model
per fold, and reports the best test epoch using standard supervised metrics.
It does not create new splits. The `train/train.py` commands below retain the
separate original-data KFold workflow.

## Independent development copy

This is version eighteen, copied from version sixteen. Its Python module name
is `src.ourexpv18` (matching the directory spelling).
Imports, default data paths, and default output paths resolve within this copy.
The default classifier is the custom model; the copied GATv2 architecture is
preserved and selected with `--model gat`. Historical comparisons below retain
their original version labels.
The original-data trainer can also discover the repository's shared raw MATLAB
files; use `--data-dir` to choose a different location.

Copied `datasyn/` artifacts and `output/` results retain their original provenance
and hashes. Paths mentioning earlier versions inside historical metadata describe
where those artifacts were produced; they are not runtime dependencies. Existing
results are inherited GATv2 results, not new version-eighteen runs. New classifier runs
read this copy's data and write into this copy's `output/` directory.

For synthetic-augmented KFold training, use the version-eighteen command in
[datacustom/README.md](datacustom/README.md). For diffusion training, use
[datasetsynthetic/eegdiffuser/README.md](datasetsynthetic/eegdiffuser/README.md).
The `temp/` reference code is not imported by the active pipelines.

## What version eighteen changes

Earlier versions scored at chance on leave-one-subject-out (LOSO): 0.499 to
0.512 for versions 10 to 17, and 0.498 for the first version-eighteen run with
the custom model on the old graphs. The cause was the data given to the
classifier, not the classifier:

| Problem found | Evidence | Change |
| --- | --- | --- |
| Half of each trial is recorded before the cue | CSP + LDA on seconds 0-2 scores 0.52 within subject; on seconds 2-4 it scores 0.79 | Only seconds 2-4 are analysed (`preprocessing.window_seconds`) |
| Raw band powers differ up to 15 times between subjects | The custom model could not fit even the training subjects (0.50) | The model takes the log and standardises each band across the electrodes of a trial (`input_norm`) |
| The graph kept only which PLV values pass 0.35 | A linear classifier on per-trial band power plus PLV values scores 0.69 on LOSO | Every electrode pair is an edge and carries its PLV as edge weight (`graphs.mode="complete"`), which the custom model reads |
| Two of the eight bands are empty | The 8-30 Hz filter leaves about zero power above 32 Hz | Six bands, 8 to 32 Hz |
| Training batches were not shuffled | The saved LOSO order groups trials by subject | `train_shuffle=True`, plus weight decay 0.001 |

Model settings (two layers, hidden size 32, dropout 0.3, standardised edge
weights) and the LOSO schedule (30 epochs, batch 32) were chosen by holding out
one training subject at a time; the LOSO test subjects were not used for that.
The window and the feature types were chosen from classical analyses that did
use all eight subjects, so LOSO results are mildly optimistic in that respect.

`legacy_dataset_config()` in [dataset/config.py](dataset/config.py) returns the
earlier rules (whole trial, eight bands, thresholded graph). The GATv2 baseline
of earlier versions corresponds to those rules with `model="gat"`,
`train_shuffle=False`, and `weight_decay=0`.

### LOSO result

One run with the version-eighteen defaults (30 epochs, batch 32, one model seed,
final-epoch evaluation, 1708 training graphs per fold including 700 synthetic):
`Version18B_GritGPS_PLV_LOSO_Synthetic_30epochs_Batch32_20261003T095309Z.json` in the external `ourexpv18/outputs/` folder.

| Held-out subject | Custom model, old graphs | Version-eighteen defaults |
| --- | ---: | ---: |
| S1 | 0.528 | 0.653 |
| S2 | 0.514 | 0.562 |
| S3 | 0.451 | 0.583 |
| S5 | 0.507 | 0.500 |
| S6 | 0.472 | 0.632 |
| S7 | 0.500 | 0.507 |
| S8 | 0.514 | 0.785 |
| S9 | 0.500 | 0.806 |
| **Mean** | **0.498** | **0.628** |

724 of 1152 held-out trials are correct (macro F1 0.596). Sampling noise on
1152 trials is about ±0.015, so this is above chance; earlier versions were not.
Subjects 5 and 7 remain at chance. For reference, CSP + LDA with per-subject
Euclidean alignment reaches 0.77 on the same window, so the graph features
still lose information that the raw signals contain.

## Reference comparison: version 10 versus version 14

This records the completed comparison used as a reference for v15 development.
These are historical v10/v14 results, not results from the new v15 generator.
Both runs contain eight subjects with ten within-subject folds each.

Sources (repository-relative links; not runtime dependencies):

- [Version 10: original-only PLV results](../ourexperimentversionten/output/BCI_IV_2a_GAT_PLV_Results.json)
- [Version 14: original plus fold-specific synthetic PLV results](../ourexperimentversionforteen/output/PLV_SavedWithinSubject_FoldSpecificSynthetic_100epochs_Batch8.json)

| Subject | Version 10 accuracy | Version 14 accuracy | Change (percentage points) |
| --- | ---: | ---: | ---: |
| S1 | 75.14% | 77.86% | +2.71 |
| S2 | 73.62% | 76.38% | +2.76 |
| S3 | 72.29% | 75.62% | +3.33 |
| S5 | 72.90% | 75.05% | +2.14 |
| S6 | 72.19% | 75.00% | +2.81 |
| S7 | 74.43% | 72.86% | -1.57 |
| S8 | 75.05% | 75.24% | +0.19 |
| S9 | 79.90% | 79.86% | -0.05 |
| **Mean** | **74.44%** | **75.98%** | **+1.54** |

| Metric | Version 10 | Version 14 | Change (percentage points) |
| --- | ---: | ---: | ---: |
| Accuracy | 74.44% | 75.98% | +1.54 |
| Balanced accuracy | 69.51% | 75.82% | +6.32 |
| Macro F1 | 67.23% | 74.19% | +6.96 |

Values average fold scores within each subject, then average across subjects.
Changes are calculated before rounding. Six subjects improved; S3 gained the
most, S7 declined, and S9 was nearly unchanged.

Version 14 used 100 classifier epochs, batch size 8, Adam at 0.001, and 100
fold-specific synthetic trials added to 129–130 original training trials per
fold. Each test fold contained 14–15 original trials. Its generators excluded
that fold's original test trials from both fitting and internal validation.

This comparison does not isolate the effect of augmentation: original fold
membership differs, and v10 selected epochs reach 245 while v14 ran for 100.
The v10 results JSON does not record its full training configuration. Both runs
select the best epoch using test-fold accuracy, so their scores are not
independent final test estimates. The numerical improvement is not a claim of
statistical significance or evidence about the new v15 generator.

For a controlled v15 comparison, evaluate original-only, original plus v14
synthetic, and original plus v15 synthetic conditions on identical original
folds with identical classifier settings and equal synthetic sample counts.
Repeat across seeds. When improving the classifier later, compare both standard
and improved classifiers under each data condition to separate generator and
classifier contributions.

## Install and run

From the repository root, the within-subject KFold experiment. Earlier versions
completed this run with the GATv2 model and the legacy graphs (74.73% mean
accuracy across subjects, selecting the best test epoch). The command below
trains the custom model on the version-eighteen graphs; that run has not been
completed yet:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv18/requirements.txt
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
python -u -m src.ourexpv18.train.train \
  --data-dir data/kaggle/bci-competition-iv-data-sets-2a \
  --subjects 1 2 3 5 6 7 8 9 \
  --folds 10 \
  --epochs 249 \
  --device cuda \
  --output "src/ourexpv18/output/PLV_035_249epochs_rerun_${run_stamp}.json"
```

This assumes the repository's `.venv` already exists and CUDA is available.
The dataset configuration uses PLV on every electrode pair. For data stored
elsewhere, change `--data-dir`. The timestamp gives each run a new output file.
The completed GATv2 run was saved as `PLV_035_249epochs_rerun_20260921T031259Z.json`
by an earlier version; that file is not part of this copy's `output/`.
Exact scores may vary between runs.

Use Python 3.10 or newer. From this folder (including a standalone copy):

```bash
python -m pip install -r requirements.txt
python train/train.py --data-dir /path/to/data
```

Provide the MATLAB files separately. Default subjects require `A01T.mat`,
`A02T.mat`, `A03T.mat`, `A05T.mat`, `A06T.mat`, `A07T.mat`, `A08T.mat`, and
`A09T.mat`. Subject 4 is excluded by default. The loader expects a MATLAB `data`
structure containing session records with signals, event positions, and labels.
Only the training recordings are used for within-subject cross-validation.

A short CPU run with real data:

```bash
python train/train.py --data-dir /path/to/data \
  --subjects 1 --folds 2 --epochs 1 --device cpu --output output/smoke.json
```

`python -m train.train` works from this folder. From the repository root, use
`python -m src.ourexpv18.train.train` with the same options.

| Option | Default |
| --- | --- |
| `--data-dir` | Automatic discovery below |
| `--output` | `output/BCI_IV_2a_GritGPS_PLV_Complete_Results.json` inside this folder (`GAT` instead of `GritGPS` with `--model gat`) |
| `--subjects` | `1 2 3 5 6 7 8 9` |
| `--folds` | `10` |
| `--epochs` | `249` per fold |
| `--device` | `auto`: CUDA when available, otherwise CPU |
| `--model` | `custom`; `gat` selects the earlier GATv2 model |

Without `--data-dir`, discovery checks this folder's
`data/bci-competition-iv-data-sets-2a`, then
`/kaggle/input/bci-competition-iv-data-sets-2a`. When located under `src/`,
it also checks the repository's `data/kaggle/bci-competition-iv-data-sets-2a`.
Explicit relative paths use the current working directory.

## Training configuration

For fold-first EEG signals and protocol-specific generation, use
`datasetsynthetic.prepare` and `datasetsynthetic.generate`. New data lives in
`datasyn/fold_first/`. The synthetic classifier consumes its completed
`protocols/within_subject.csv`; it does not use the copied historical partitions.
See [datacustom/README.md](datacustom/README.md) for graph loading and training.

Edit [train/config.py](train/config.py) for training defaults and
[dataset/config.py](dataset/config.py) for signal and graph settings.
Python callers can customize the frozen configurations with `dataclasses.replace`
and call `train.experiment.run_experiment`.

Each subject has separate shuffled KFold splits (seed 42). Each fold creates a
fresh model (seed 12345) and Adam optimizer with learning rate 0.001. Batch size
is 32; the training loader shuffles and the test loader does not; weight decay
is 0.001. Defaults use 249 epochs and the custom model with the sizes in
[model/custom/config.py](model/custom/config.py): hidden size 32, two layers,
four attention heads, dropout 0.3, attention dropout 0.2.
`hidden_channels` (22) and `heads` (3) in `train/config.py` apply to the GATv2
model only, which also uses dropout 0.5.

The LOSO trainer (`train/train_syn_loso.py`) evaluates each held-out subject
once at the final epoch and defaults to 30 epochs at batch 32.

The epoch with highest test-fold accuracy is selected; ties retain the earliest
epoch. This preserves the existing supervised behavior. There is no separate
validation split, early stopping, or checkpoint resume.

## Pipeline How EEG becomes a prediction

| Stage | Current behavior |
| --- | --- |
| Select trials | Session indices 3–8 and channel indices 0–21, both zero-based |
| Select classes | Original labels 1 and 2 become targets 0 (left) and 1 (right); other labels are skipped |
| Extract signal | Four seconds at 250 Hz: 1,000 samples × 22 channels per complete trial |
| Filter | Fifth-order Butterworth, zero-phase, 8–30 Hz |
| Select window | Seconds 2–4 of the filtered trial (500 samples); the cue appears 2 s after trial start |
| Compute connectivity | Hilbert phases give PLV for each electrode pair; diagonal defaults to zero |
| Compute node features | Six bandpowers: 8–12, 12–16, 16–20, 20–24, 24–28, 28–32 Hz |
| Build graph | Every electrode pair is an edge (462 directed edges) with its PLV as edge weight |
| Predict | Custom model: log and per-trial standardisation of the band powers, then two layers, each with a GatedGCN branch over the graph edges and GRIT attention over all electrode pairs, then mean pooling of nodes and pairs and a two-class MLP. `--model gat`: three GATv2 layers, GraphNorm, global mean pooling, dropout, and a two-class linear output |

PLV is `abs(mean(exp(1j * phase_difference)))` over the trial's time samples.
It gives a symmetric connection matrix. Only `method="plv"` is accepted.

For features, each band receives its own fifth-order filter, followed by Welch
power estimation and Simpson integration. Welch defaults are a two-second Hann
window, 50% overlap, and constant detrending. Band boundaries are inclusive.
**The initial 8–30 Hz filter is applied before all feature bands**, so these are
not bandpowers computed directly from unfiltered EEG. The dataset code does not
standardize features; the custom model does it per trial.

Each graph contains `x` of shape `[22, 6]`, `edge_index` of shape `[2, 462]`,
`edge_weight` of shape `[462]`, one target `y`, and subject/session/trial IDs. The input feature count and the
electrode count are inferred from the graphs. The custom model adds a learned
embedding per electrode, so every graph must list the same electrodes in the
same order. It drops self-loops and duplicate edges, which removes the legacy
`(0, 0)` placeholders. The custom model reads the edge weights whenever the
graphs carry them; GATv2 uses the topology only.

### Dataset details that affect comparisons

- Event positions are converted with `int(position) + event_index_offset`;
  the default offset is zero. No automatic one-based index correction occurs.
- Incomplete trials are skipped by default. Set `incomplete_policy="error"`
  to reject them instead. Trials are not automatically padded.
- `trials.order="legacy_interleave"` interleaves two halves of the class-sorted
  trial list. It drops the final item when the total is odd, and does not ensure
  alternating classes when class counts differ. `"interleave"` alternates actual
  class groups and retains every trial; `"grouped"` keeps class-sorted order.
- `graphs.mode="complete"` (default) keeps every electrode pair without
  self-loops and ignores the threshold.
- `graphs.mode="legacy"` first uses a strict PLV threshold, then a `>=` pass.
  Rejected pairs become `(0, 0)` placeholders instead of disappearing. At
  threshold zero, all node pairs, including the diagonal, are retained.
- `graphs.mode="threshold"` omits rejected edges and supports `comparison="gt"`
  or `"ge"`, self-loop selection, and direction settings. Undirected graphs
  emit both edge orientations. These settings govern stored edges; GATv2 layers
  use their own default self-loop handling and the custom model ignores
  self-loops.

These rules are deliberate parts of the pipeline; changing them changes the
experiment.

## Results

The runner writes one JSON after all subjects finish, containing accuracy,
balanced accuracy, macro F1, confusion matrices, and selected epochs per fold.
It does not save model weights or per-epoch history. An existing output path is
overwritten. Files already in `output/` are historical results.

The top-level JSON keys are subject IDs as strings, such as `"1"`. Each subject
has these fields:

| Field | Meaning |
| --- | --- |
| `mean`, `max`, `min` | Accuracy summary across the selected epochs of all folds |
| `balanced_accuracy` | Mean/max/min of fold balanced accuracies |
| `macro_f1` | Mean/max/min of fold macro F1 scores |
| `confusion_matrix` | Sum of the selected-epoch confusion matrices across folds |
| `class_labels` | `[0, 1]` |
| `folds` | Each fold's number, selected epoch, accuracy, balanced accuracy, macro F1, and confusion matrix |

Scores are fractions: `0.75` means 75%. Fold summaries use unweighted averages.
Balanced accuracy averages recall across classes present in the true labels.
Macro F1 uses both target classes and gives zero to undefined class scores.
Confusion matrices always have two rows and columns: rows are true classes,
columns are predictions, in order `[left, right]`. Their entries count trials.
All metrics in a fold come from the same selected epoch.

**Interpretation of the scores:** the trainer evaluates the test fold every
epoch and selects the epoch with its highest accuracy. The reported test scores
therefore also influenced epoch selection; they are not an independent final
evaluation. There is no separate validation split. The first epoch is selected
even if all epochs have zero accuracy, and later ties keep the earlier epoch.

## Verify

From this folder:

```bash
python -m unittest discover -s tests -v
python train/train.py --help
```

Tests use synthetic EEG and bundled regression fixtures, including the original
supervised training results. The real dataset is not required for these checks.

### Saved within-subject folds with synthetic augmentation

The [synthetic-training command](datacustom/README.md#run-supervised-training)
uses the completed `datasyn/fold_first/` bundle with 100 classifier epochs and
batch size 8. Each fold uses its own generated signals. Reports go to this
version's `output/`. Cross-subject loading will be added separately.
