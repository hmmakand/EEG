# Version fifteen: standalone supervised EEG classification

This folder contains the dataset pipeline, GATv2 model, and standard supervised
training for left-hand versus right-hand motor imagery from BCI Competition IV 2a.
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

This is version fifteen, copied from version fourteen. Its Python module name
is `src.ourexpv15` (matching the directory spelling).
Imports, default data paths, and default output paths resolve within this copy.
The original-data trainer can also discover the repository's shared raw MATLAB
files; use `--data-dir` to choose a different location.

Copied `datasyn/` artifacts and `output/` results retain their original provenance
and hashes. Paths mentioning earlier versions inside historical metadata describe
where those artifacts were produced; they are not runtime dependencies. Existing
results are inherited results, not new version-fifteen runs. New classifier runs
read this copy's data and write into this copy's `output/` directory.

For synthetic-augmented KFold training, use the version-fifteen command in
[datacustom/README.md](datacustom/README.md). For diffusion training, use
[datasetsynthetic/eegdiffuser/README.md](datasetsynthetic/eegdiffuser/README.md).
The `temp/` reference code is not imported by the active pipelines.

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

From the repository root, run with the settings used for the completed
249-epoch CUDA experiment (74.73% mean accuracy across subjects):

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv15/requirements.txt
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
python -u -m src.ourexpv15.train.train \
  --data-dir data/kaggle/bci-competition-iv-data-sets-2a \
  --subjects 1 2 3 5 6 7 8 9 \
  --folds 10 \
  --epochs 249 \
  --device cuda \
  --output "src/ourexpv15/output/PLV_035_249epochs_rerun_${run_stamp}.json"
```

This assumes the repository's `.venv` already exists and CUDA is available.
The dataset configuration uses PLV with graph threshold `0.35`. For data stored
elsewhere, change `--data-dir`. The timestamp gives each run a new output file.
The completed run is saved in
[PLV_035_249epochs_rerun_20260921T031259Z.json](output/PLV_035_249epochs_rerun_20260921T031259Z.json).
The command uses the same training settings; exact scores may vary between runs.

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
`python -m src.ourexpv15.train.train` with the same options.

| Option | Default |
| --- | --- |
| `--data-dir` | Automatic discovery below |
| `--output` | `output/BCI_IV_2a_GAT_PLV_Threshold_0.35_Results.json` inside this folder |
| `--subjects` | `1 2 3 5 6 7 8 9` |
| `--folds` | `10` |
| `--epochs` | `249` per fold |
| `--device` | `auto`: CUDA when available, otherwise CPU |

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
is 32; training and test loaders do not shuffle. Defaults use 249 epochs,
22 hidden channels per head, three attention heads, and dropout 0.5.

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
| Compute connectivity | Hilbert phases give PLV for each electrode pair; diagonal defaults to zero |
| Compute node features | Eight bandpowers: 8–12, 12–16, 16–20, 20–24, 24–28, 28–32, 32–36, 36–40 Hz |
| Build graph | Apply graph threshold `0.35`, using the legacy edge construction by default |
| Predict | Three GATv2 layers, GraphNorm, global mean pooling, dropout, and a two-class linear output |

PLV is `abs(mean(exp(1j * phase_difference)))` over the trial's time samples.
It gives a symmetric connection matrix. Only `method="plv"` is accepted.

For features, each band receives its own fifth-order filter, followed by Welch
power estimation and Simpson integration. Welch defaults are a two-second Hann
window, 50% overlap, and constant detrending. Band boundaries are inclusive.
**The initial 8–30 Hz filter is applied before all eight feature bands**, including
those above 30 Hz. This is preserved behavior, so these are not eight bandpowers
computed directly from unfiltered EEG. Features are not standardized by this code.

Each graph contains `x` of shape `[22, 8]`, `edge_index` of shape `[2, edges]`,
one target `y`, and subject/session/trial IDs. Model defaults are 22 hidden
channels per attention head, three heads, and dropout probability 0.5.
The input feature count is inferred from the graphs. Edge weights, even if
attached to a graph, are not consumed by the current model.

### Dataset details that affect comparisons

- Event positions are converted with `int(position) + event_index_offset`;
  the default offset is zero. No automatic one-based index correction occurs.
- Incomplete trials are skipped by default. Set `incomplete_policy="error"`
  to reject them instead. Trials are not automatically padded.
- `trials.order="legacy_interleave"` interleaves two halves of the class-sorted
  trial list. It drops the final item when the total is odd, and does not ensure
  alternating classes when class counts differ. `"interleave"` alternates actual
  class groups and retains every trial; `"grouped"` keeps class-sorted order.
- `graphs.mode="legacy"` first uses a strict PLV threshold, then a `>=` pass.
  Rejected pairs become `(0, 0)` placeholders instead of disappearing. At
  threshold zero, all node pairs, including the diagonal, are retained.
- `graphs.mode="threshold"` omits rejected edges and supports `comparison="gt"`
  or `"ge"`, self-loop selection, and direction settings. Undirected graphs
  emit both edge orientations. These settings govern stored edges; GATv2 layers
  use their own default self-loop handling.

These compatibility defaults are deliberate parts of the current pipeline;
changing them changes the experiment.

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
