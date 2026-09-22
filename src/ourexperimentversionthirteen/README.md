# Version thirteen: standalone supervised EEG classification

This folder contains the dataset pipeline, GATv2 model, and standard supervised
training for left-hand versus right-hand motor imagery from BCI Competition IV 2a.
Copy the whole folder anywhere; no other experiment or repository files are needed.
Training uses labeled trial graphs, Adam, and cross-entropy loss.

For **fixed train/validation/test partitions**, with optional synthetic training
trials, use the new `train/train_syn.py` workflow described in
[datacustom/README.md](datacustom/README.md). It builds graphs in memory, selects
epochs using validation, and tests once. The `train/train.py` commands below
retain the existing KFold workflow.

## Install and run

From the repository root, run with the settings used for the completed
249-epoch CUDA experiment (74.73% mean accuracy across subjects):

```bash
source .venv/bin/activate
python -m pip install -r src/ourexperimentversionthirteen/requirements.txt
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
python -u -m src.ourexperimentversionthirteen.train.train \
  --data-dir data/kaggle/bci-competition-iv-data-sets-2a \
  --subjects 1 2 3 5 6 7 8 9 \
  --folds 10 \
  --epochs 249 \
  --device cuda \
  --output "src/ourexperimentversionthirteen/output/PLV_035_249epochs_rerun_${run_stamp}.json"
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
`python -m src.ourexperimentversionthirteen.train.train` with the same options.

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

For original EEG segments with fixed 70/15/15 training/validation/testing splits,
see [datasetsynthetic/README.md](datasetsynthetic/README.md). That separate
preparation command saves reusable partitions under `datasyn/original/`, without
filtering, augmentation, or graph creation. The existing trainer below still
uses its original cross-validation procedure.

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

### Original-only folds with synthetic augmentation

Use `train/train_syn.py --protocol kfold` for within-subject folds formed from
all 144 originals, with existing synthetic trials added to every training fold.
The [command and protocol](datacustom/README.md#within-subject-folds-with-existing-synthetic-data)
cover 100 epochs, batch size 8, output naming, and the known synthetic leakage
and best-test epoch-selection limitations. The standard trainer is unchanged.
