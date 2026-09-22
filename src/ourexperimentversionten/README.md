# Version ten: standalone EEG experiment -- this one is baseline for py script project with bci-iv-plv

This folder contains its own dataset preprocessing, GATv2 model, and training
script. You can copy it outside this repository; no other experiment or Git
checkout is required. Python 3.10 or newer is required.

From this folder, install the dependencies and run:

```bash
python -m pip install -r requirements.txt
python train/train.py --data-dir /path/to/bci-competition-iv-data-sets-2a
```

The data directory must contain the BCI Competition IV 2a MATLAB training files
`A01T.mat`, `A02T.mat`, `A03T.mat`, `A05T.mat`, `A06T.mat`, `A07T.mat`,
`A08T.mat`, and `A09T.mat`, with the `data` structure expected by the loader.
Data is supplied separately and is not downloaded automatically.

Without `--data-dir`, the script checks `data/bci-competition-iv-data-sets-2a`
inside this folder, then `/kaggle/input/bci-competition-iv-data-sets-2a`, then
the existing repository's `data/kaggle/bci-competition-iv-data-sets-2a` location
when this folder is under `src/`. Paths do not depend on your working directory.

Results default to `output/BCI_IV_2a_GAT_COHERENCE_Results.json` for coherence.
Other methods use the same pattern with `PLV`, `PLI`, or `WPLI` inside this folder.
The filename follows `ConnectivityConfig.method` automatically.
Use `--output /path/to/results.json` to choose another destination.

From the repository root, module execution is also supported:

```bash
python -m src.ourexperimentversionten.train.train --data-dir /path/to/data
```

The experiment uses the original training settings:
subjects 1–3 and 5–9, 10-fold cross-validation, batch size 32, and
`num_epochs=249` (the original `range(1, 250)` executed 249 epochs per fold).
CUDA is used when available.
Run `python train/train.py --help` for path options.

## Dataset modules and rule ownership

| Module | Rules and main functions |
| --- | --- |
| `dataset/config.py` | Immutable stage settings in `DatasetConfig`; `default_dataset_config()` |
| `dataset/data_io.py` | Search locations, filenames, MATLAB decoding: `resolve_data_dir`, `subject_file_path`, `load_mat_file`, `read_sessions` |
| `dataset/trials.py` | Sessions, channels, event offset, duration, incomplete trials, labels, ordering: `select_sessions`, `select_channels`, `event_to_sample`, `extract_trials`, `encode_label`, `order_trials` |
| `dataset/preprocessing.py` | Butterworth SOS zero-phase filtering and time axis: `design_bandpass`, `apply_bandpass`, `preprocess_trial`; optional `pad_trials` |
| `dataset/connectivity.py` | PLV/PLI/WPLI and band-averaged coherence: `extract_phase`, `compute_plv_matrix`, `compute_pli_matrix`, `compute_wpli_matrix`, `compute_coherence_matrix`, `compute_connectivity` |
| `dataset/features.py` | Band ranges, per-band filtering, Welch parameters, Simpson integration: `make_band_ranges`, `compute_psd`, `integrate_bandpower`, `compute_node_features` |
| `dataset/graphs.py` | Threshold comparison, self-loops, edge direction/weights, tensor types: `make_edge_mask`, `connectivity_to_edges`, `make_pyg_graph` |
| `dataset/validation.py` | Boundary checks: `validate_config`, `validate_session`, `validate_trial`, `validate_graph` |
| `dataset/dataset.py` | Execution order: `build_trial_graph`, `build_subject_dataset` |

The training pipeline calls `build_subject_dataset()` and derives the model input
feature count from its graphs. Dataset defaults live in `dataset/config.py`;
training defaults live in `train/config.py`. Both configurations are passed
explicitly into the experiment functions.
The old monolithic helper API is replaced by the functions in the table above.

To change rules without changing their implementations, use `dataclasses.replace`:

```python
from dataclasses import replace
from dataset import build_subject_dataset, default_dataset_config

config = default_dataset_config()
config = replace(
    config,
    trials=replace(config.trials, channel_indices=(0, 1, 2), order="interleave"),
    features=replace(config.features, band_edges=(8, 12, 16, 20)),
    graphs=replace(config.graphs, mode="threshold", threshold=0.5),
)
graphs = build_subject_dataset("/path/to/data", 1, config)
```

Each `Trial` carries `[samples, channels]` signals, its target label, subject,
session, and trial IDs. Graphs retain these IDs, with `x` shaped
`[channels, bands]`, `edge_index` shaped `[2, edges]`, and a scalar integer `y`.
The loader extracts equal-duration trials, so the pipeline no longer needs the
original class-wise padding or duplicated band arrays. `pad_trials()` is available
for callers handling variable-duration trials; it is not applied automatically.

### Preserved defaults and optional rule changes

- Sessions 3–8, channels 0–21, source classes 1/2 mapped to 0/1, 250 Hz,
  four-second trials, and event offset zero remain the defaults. Event index
  conventions have not been changed or reinterpreted.
- Every feature band still receives the initial 8–30 Hz filtering first.
- The default graph threshold is `0.35`, configured in `dataset/config.py`.
- `graphs.mode="legacy"` preserves the original strict adjacency threshold,
  followed by `>=` selection and `(0, 0)` placeholders for rejected entries.
  Threshold zero therefore includes all node pairs and the diagonal. Custom
  comparison, self-loop, and direction settings require `mode="threshold"`.
- `graphs.mode="threshold"` omits rejected edges. `comparison="gt"` or `"ge"`
  selects the comparison; `self_loops` allows qualifying diagonal entries.
  Undirected graphs emit both edge orientations. `include_edge_weights=True`
  attaches weights to the data, but the current GAT does not consume them.
- `trials.order="legacy_interleave"` preserves the original half-split ordering,
  including dropping the final item for odd totals. `"interleave"` instead
  alternates actual class groups and retains all trials; `"grouped"` retains
  the class-grouped extraction order.
- The dataset supports configurable label mappings. The current trainer/model
  is binary and rejects mappings with other than two targets.

### Verification

From this folder:

```bash
python -m unittest discover -s tests -v
```

The regression fixture compares features, labels, and edge indices against
outputs captured from the original pipeline before the refactor, at thresholds
0 and 0.1. Additional checks cover configurable channels/bands, trial rules,
unequal class counts, graph selection, model backpropagation, and standalone CLI
execution. Full experiment training is not required by these checks.


## Training modules and rule ownership

| Module | Rules and functions |
| --- | --- |
| `train/config.py` | Subjects, folds, seeds, batches, learning rate, epochs, model settings, device, output: `TrainingConfig`, `default_training_config`, `validate_training_config` |
| `train/setup.py` | Device and initialization, GAT construction, Adam, cross-entropy: `select_device`, `seed_fold`, `build_model`, `build_optimizer`, `build_criterion` |
| `train/splitting.py` | KFold partitions and batch loaders: `make_fold_indices`, `select_fold_data`, `build_loaders` |
| `train/engine.py` | Gradient updates, accuracy, and optional prediction collection: `train_epoch`, `evaluate` |
| `train/selection.py` | Highest test accuracy with strict improvement: `initialize_selection`, `update_selection`, `finalize_selection` |
| `train/metrics.py` | Classification metrics and fold summaries: `compute_classification_metrics`, `summarize_fold_scores`, `summarize_fold_results` |
| `train/experiment.py` | Execution order: `run_fold`, `run_subject`, `run_experiment` |
| `train/reporting.py` | Console output and JSON saving: `print_subject_result`, `print_summary`, `save_results` |
| `train/train.py` | CLI and input-file checks: `parse_args`, `validate_input_files`, `main` |

The default loop executes exactly `num_epochs` iterations. Its default 249
preserves the old loop's behavior. Split shuffling (seed 42), the fold seed reset
(12345), nonshuffled loaders, batch size 32, Adam at 0.001, and per-epoch train/test
accuracy evaluation are unchanged. Epoch selection still uses the highest test
accuracy and keeps the earlier selection on ties; it does not save model weights.
The unused average of train/test accuracy has been removed. It never influenced
epoch selection or saved results.

Each subject retains the original top-level accuracy `mean`, `max`, and `min`.
Additional results include balanced accuracy, macro F1, a confusion matrix, and
per-fold details, described below. JSON serializes subject IDs as strings.
The graph threshold remains 0.35 in the dataset configuration only.

From this folder, a custom run can pass configurations without changing globals:

```python
from dataclasses import replace
from dataset import default_dataset_config
from train.config import default_training_config
from train.experiment import run_experiment
from train.reporting import print_summary, save_results, resolve_results_path
from train.setup import select_device

training = replace(default_training_config(), subjects=(1,), n_folds=2, num_epochs=3)
dataset = default_dataset_config()
results = run_experiment(
    data_dir="/path/to/data",
    device=select_device(training.device),
    dataset_config=dataset,
    training_config=training,
)
print_summary(results)
save_results(results, resolve_results_path(dataset.connectivity.method, training.output_path))
```

The former `train.train.run_subject` function now lives in `train.experiment`
and takes both configurations explicitly. The CLI commands above remain valid.
To change optimizer, loss, or selection algorithms, extend the corresponding
function and configuration validation; unsupported names raise an error.

Training checks compare fold indices, per-epoch accuracies, and mean/max/min
against a CPU run captured before the training refactor (two folds, three epochs,
threshold 0.35). Tests also cover initial parameters, tie handling, sample-weighted
accuracy, CLI result saving, and standalone and repository module execution.


## Additional classification metrics

`sklearn.metrics` calculates balanced accuracy, macro F1, and the confusion
matrix from all test predictions for the selected epoch of each fold. Predictions
are collected during the existing test evaluation; there is no extra model pass.
Selection still uses highest test accuracy, retaining the earlier epoch on ties.
The first epoch supplies metrics even if every epoch has zero accuracy.

- Balanced accuracy averages recall across classes present in the fold's true
  labels, following scikit-learn's definition. If a class is absent, it does not
  contribute to this average.
- Macro F1 uses fixed target labels `[0, 1]` and `zero_division=0`, so both classes
  have equal weight and undefined class scores contribute zero.
- Confusion matrices always have shape 2 by 2: rows are true labels and columns
  are predicted labels. With the default mapping, 0 is left and 1 is right.
- Balanced accuracy and macro F1 are summarized with an unweighted mean, maximum,
  and minimum across folds, just like existing accuracy. The subject confusion
  matrix is the **sum** of selected-epoch test matrices across folds; it counts
  trials, not averaged percentages.

Each subject's saved result has this structure:

```text
mean, max, min             existing selected test accuracy summary
balanced_accuracy         {mean, max, min} across folds
macro_f1                  {mean, max, min} across folds
confusion_matrix          summed 2 by 2 counts
class_labels              [0, 1]
folds                     list of fold, selected_epoch, accuracy,
                          balanced_accuracy, macro_f1, confusion_matrix
```

Fold and epoch numbers start at 1. All metrics in a fold come from the same
selected epoch; each metric is not independently maximized. These results retain
the experiment's existing best-test-epoch selection protocol.


## Switching connectivity

`ConnectivityConfig.method` in `dataset/config.py` now defaults to `"coherence"`.
Change it to `"plv"`, `"pli"`, or `"wpli"` to select another supported method.
No trainer or model changes are required.

The existing PLI implementation computes
`abs(mean(sign(sin(phase_difference))))` over time samples in each filtered
trial, using Hilbert phases. This is an absolute, symmetric PLI matrix; the
diagonal defaults to zero. It is not weighted PLI or directed PLI. See the
[MNE discussion of PLI](https://mne.tools/mne-connectivity/stable/auto_examples/dpli_wpli_pli.html)
for the sign-based definition; this pipeline averages Hilbert phase relationships
within each trial rather than estimating cross-spectra across epochs.

The graph threshold remains `0.35` and graph mode remains `"legacy"`. Changing
connectivity methods changes values, so the threshold is not automatically calibrated
to the new method. The historical regression fixtures explicitly select PLV;
ordinary dataset and short training checks exercise the coherence default.


### Coherence settings

`ConnectivityConfig` supplies these coherence-only defaults:

```python
method: str = "coherence"
fmin: float = 8.0
fmax: float = 30.0
nperseg: int = 256
noverlap: int | None = None  # SciPy uses nperseg // 2.
window: str = "hann"
detrend: str = "constant"
```

`dataset.py` passes `DatasetConfig.sample_rate` (250 Hz by default) to
`compute_connectivity`. Coherence uses the existing filtered time-series signal,
then averages magnitude-squared coherence equally over the frequency bins in
`[fmin, fmax]`. It returns one symmetric matrix per trial with the configured
diagonal (default zero). Thresholding remains in `graphs.py` at 0.35.

Validation applies coherence settings only when coherence is selected. It checks
the band against Nyquist, segment size and overlap, window and detrending rules,
at least two complete Welch segments, and at least one frequency bin in the
selected band. The estimator also checks the actual signal length and rejects
constant channels or nonfinite estimates rather than silently generating edges
from undefined values. PLV/PLI/WPLI dispatch remains available without a sampling
rate argument; coherence requires it explicitly.

Tests cover identical/independent signals, symmetry and range, comparison with
SciPy for a custom sampling rate and band, invalid settings/signals, and the
method-specific output filename. Full training is not part of these checks.
