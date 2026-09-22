# Version twelve: EEG graph classification

This experiment classifies **left-hand versus right-hand motor imagery** from
BCI Competition IV 2a EEG recordings. Each four-second trial becomes a graph:
electrodes are nodes, bandpower values are node features, and PLV determines
connections. A GATv2 neural network predicts one class for each trial.

The current code supports **PLV connectivity** and two training engines:
**supervised** and **FixMatch-style training**. VAT scripts are no longer present.
Files in `output/` may come from older implementations, including VAT and other
connectivity methods; their presence does not mean those methods are runnable now.

## 1. Install and locate the data

Use Python 3.10 or newer. From the repository root:

```bash
# Activate the repository environment if it already exists.
source .venv/bin/activate
python -m pip install -r src/ourexperimentversiontwelve/requirements.txt
```

All shell commands below assume the repository root and an active Python
environment. If copying this experiment elsewhere, install its `requirements.txt`
and run `python train/train.py` or `python train/train_fixmatch.py` from the copied
folder. No other experiment folder is needed at runtime.

Provide the MATLAB training files separately; the code does not download data.
The default subject list needs these files in one directory:

```text
A01T.mat  A02T.mat  A03T.mat  A05T.mat
A06T.mat  A07T.mat  A08T.mat  A09T.mat
```

The loader expects a MATLAB `data` structure containing session records with
signals, event positions, and labels. Subject 4 is excluded by the configured
subject list. This runner uses the `T.mat` recordings for within-subject
cross-validation; it does not evaluate separate `E.mat` recordings.

Pass `--data-dir /path/to/data` to choose the directory. Without it, the loader
uses the first existing directory in this order:

1. `src/ourexperimentversiontwelve/data/bci-competition-iv-data-sets-2a`
2. `/kaggle/input/bci-competition-iv-data-sets-2a`
3. `data/kaggle/bci-competition-iv-data-sets-2a` under the repository root

These automatic locations are resolved from the script location. Explicit
relative `--data-dir` and `--output` paths are relative to your working directory.

## 2. Choose a training command

### Supervised baseline

```bash
python src/ourexperimentversiontwelve/train/train.py
```

This uses labeled graphs and cross-entropy loss. Its CLI accepts `--data-dir`
and `--output`. It does **not** accept `--epochs`; use the Python example below
to customize supervised training.

### FixMatch with training graphs reused for pseudo-labeling

To run for **100 epochs per fold**:

```bash
python src/ourexperimentversiontwelve/train/train_fixmatch.py \
  --epochs 100 \
  --fixmatch-reuse-labeled \
  --output src/ourexperimentversiontwelve/output/PLV_FIXMATCH_ReuseLabeled_100epochs.json
```

The reuse flag sends the fold's training graphs through both the supervised and
pseudo-label paths. Their true labels are ignored in the pseudo-label path.
This adds no new unlabeled recordings; it tests regularization on data already
used for supervised training. Test-fold graphs are not included in that pool.

**Without `--fixmatch-reuse-labeled`, this CLI runs only supervised training
with weak feature noise.** It does not create an unlabeled split, so there is
no pseudo-label loss. A separate unlabeled pool is supported through the Python
API described in section 5.

For a short check with real data:

```bash
python src/ourexperimentversiontwelve/train/train_fixmatch.py \
  --subjects 1 --folds 2 --epochs 1 --device cpu \
  --fixmatch-reuse-labeled \
  --output src/ourexperimentversiontwelve/output/fixmatch_smoke.json
```

Add `--data-dir /path/to/data` to any command if automatic discovery is unsuitable.
Run either script with `--help` to see its supported options. Module execution
also works, for example:

```bash
python -m src.ourexperimentversiontwelve.train.train_fixmatch --epochs 100 --fixmatch-reuse-labeled
```

### FixMatch CLI options

| Option | Default | Meaning |
| --- | --- | --- |
| `--epochs` | `249` | Complete passes through the labeled training loader per fold |
| `--folds` | `10` | Cross-validation folds for each subject |
| `--subjects` | `1 2 3 5 6 7 8 9` | Space-separated subject IDs |
| `--device` | `auto` | CUDA when available, otherwise CPU; also accepts `cpu` or `cuda` |
| `--fixmatch-reuse-labeled` | Off | Use the fold's labeled training graphs as the pseudo-label pool too |
| `--fixmatch-lambda-u` | `1.0` | Weight of the pseudo-label loss |
| `--fixmatch-confidence-threshold` | `0.95` | Minimum probability required to accept a pseudo-label |
| `--fixmatch-weak-noise-std` | `0.05` | Gaussian noise standard deviation for weak views |
| `--fixmatch-strong-noise-std` | `0.2` | Gaussian noise standard deviation for strong views |
| `--fixmatch-strong-mask-prob` | `0.3` | Probability of setting each strong-view feature entry to zero |
| `--data-dir` | Automatic discovery | Directory containing the selected subjects' MATLAB files |
| `--output` | See section 6 | Destination JSON file |

The **249-epoch default** preserves the original `range(1, 250)` loop. It is not
a requirement of the model or FixMatch. Using `--epochs 100` runs exactly 100
epochs per fold. With all defaults, eight subjects each receive ten independent
fold runs. There is no early stopping or checkpoint resume.

## 3. How EEG becomes a prediction

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

## 4. Training defaults and custom configuration

Dataset settings live in [dataset/config.py](dataset/config.py). Training
settings live in [train/config.py](train/config.py).

| Setting | Default |
| --- | --- |
| Engine | `supervised` (`train_fixmatch.py` selects `fixmatch`) |
| Batch size | `32` |
| Optimizer / learning rate | Adam / `0.001` |
| Loss | Cross-entropy |
| Fold splitting | Shuffled KFold, seed `42`; not stratified or grouped by session |
| Training/test loader shuffling | Both off |
| Model seed | Reset to `12345` for every fold |
| Epoch selection | Highest test-fold accuracy; earliest epoch wins ties |

A new model and optimizer are created for every fold. Each subject is trained
separately; this is not a model trained across all subjects.

For a custom supervised run, including 100 epochs, run the following from the
repository root. The same configuration approach works for FixMatch.

```bash
python - <<'PY'
from dataclasses import replace
from pathlib import Path
from src.ourexperimentversiontwelve.dataset import default_dataset_config
from src.ourexperimentversiontwelve.dataset.data_io import resolve_data_dir
from src.ourexperimentversiontwelve.train.config import default_training_config
from src.ourexperimentversiontwelve.train.experiment import run_experiment
from src.ourexperimentversiontwelve.train.reporting import print_summary, save_results
from src.ourexperimentversiontwelve.train.setup import select_device

training = replace(default_training_config(), num_epochs=100)
dataset = default_dataset_config()
# Example graph override:
# dataset = replace(dataset, graphs=replace(dataset.graphs, mode="threshold", threshold=0.5))

results = run_experiment(
    data_dir=resolve_data_dir(None),  # Or provide an explicit dataset directory.
    device=select_device(training.device),
    dataset_config=dataset,
    training_config=training,
)
print_summary(results)
save_results(results, Path("src/ourexperimentversiontwelve/output/PLV_supervised_100epochs.json"))
PY
```

## 5. What the FixMatch engine does

[train/fixmatch_engine.py](train/fixmatch_engine.py) implements this sequence:

1. Add weak Gaussian noise to labeled features and compute supervised cross-entropy.
2. If an unlabeled loader is supplied and `lambda_u > 0`, take one unlabeled batch.
3. Predict its weakly augmented view with dropout disabled and gradients disabled.
4. Use the most probable class as a pseudo-label, accepting it only when its
   probability meets the confidence threshold.
5. Add stronger Gaussian noise and randomly zero feature entries. Predict this
   strong view in training mode, and compute cross-entropy on accepted graphs only.
6. Update the model with `supervised_loss + lambda_u * unsupervised_loss`.

The pseudo-label loss is averaged over **accepted graphs**, not the entire
unlabeled batch. If none are accepted, that update has zero pseudo-label loss.
PLV edges stay fixed for every augmented view. Noise is in the existing bandpower
units; the engine does not standardize or clip augmented features.

`lambda_u=0` disables pseudo-label work but still applies weak noise to labeled
features. Setting `weak_noise_std=0` as well removes that augmentation.

### Using separate unlabeled training graphs

The engine and `run_fold()` accept a PyG `unlabeled_loader`. Prepare that pool
explicitly from training data; the command-line runner has no unlabeled-data
argument. This Python fragment assumes you have already created the graph lists:

```python
from torch_geometric.loader import DataLoader
from src.ourexperimentversiontwelve.train.experiment import run_fold

training = replace(training, training_engine="fixmatch",
                   fixmatch_reuse_labeled_as_unlabeled=False)
unlabeled_loader = DataLoader(unlabeled_graphs, batch_size=training.batch_size)
result = run_fold(
    labeled_train_graphs, test_graphs, select_device(training.device),
    training, subject_number=1, fold_number=1,
    unlabeled_loader=unlabeled_loader,
)
```

Unlabeled graphs need node features and edges; their labels are never read.
Keep test graphs out of this loader. An explicit loader cannot be combined with
`fixmatch_reuse_labeled_as_unlabeled=True`.

One labeled epoch pairs each labeled batch with one unlabeled batch. The
unlabeled loader restarts when exhausted; if it is longer than the labeled
loader, some batches are not used in that epoch.

For direct calls, `train_epoch_fixmatch()` returns `supervised_loss`,
`unsupervised_loss`, `total_loss`, and `pseudo_label_mask_rate` (the accepted
fraction). These are epoch averages weighted by labeled graph counts.
The experiment runner currently discards these diagnostics; they are not
printed or saved in the results JSON.

## 6. Outputs and how to read them

Default files are written inside this experiment's `output/` directory:

| Entry point | Default filename |
| --- | --- |
| `train.py` | `BCI_IV_2a_GAT_PLV_Threshold_0.35_Results.json` |
| `train_fixmatch.py` | `BCI_IV_2a_GAT_PLV_FIXMATCH_Threshold_0.35_Results.json` |

The threshold portion follows the dataset configuration. The FixMatch default
name does **not** distinguish reuse mode, epoch count, or other hyperparameters.
Use `--output` to distinguish runs; an existing file at that path is overwritten.

The CLI writes one JSON after the entire experiment finishes. It does not save
model weights, checkpoints, per-epoch history, or a complete configuration.
Keep the launch command/settings alongside results if you need to reproduce a run.
Historical filenames alone cannot establish the settings used to generate them.

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

## 7. Where to change the code

| File or folder | Responsibility |
| --- | --- |
| `dataset/config.py` | Trial, filter, feature, connectivity, and graph settings |
| `dataset/data_io.py`, `dataset/trials.py` | Load MATLAB sessions, extract and order trials |
| `dataset/preprocessing.py`, `dataset/features.py` | Signal filtering and bandpower features |
| `dataset/connectivity.py`, `dataset/graphs.py` | PLV matrices and PyG graph construction |
| `dataset/dataset.py`, `dataset/validation.py` | Dataset orchestration and boundary checks |
| `model/gat.py` | GATv2 architecture |
| `train/config.py`, `train/setup.py` | Training settings, device, model and optimizer creation |
| `train/train.py`, `train/train_fixmatch.py` | Command-line entry points |
| `train/engine.py`, `train/fixmatch_engine.py` | One training epoch and evaluation |
| `train/splitting.py`, `train/experiment.py` | Loaders, folds, subjects, and complete experiments |
| `train/selection.py`, `train/metrics.py`, `train/reporting.py` | Epoch selection, metrics, display, and JSON saving |
| `tests/` | Synthetic data checks, regression fixtures, and training tests |

## 8. Verify the installation

From the repository root:

```bash
python -m unittest discover -s src/ourexperimentversiontwelve/tests -v
python src/ourexperimentversiontwelve/train/train.py --help
python src/ourexperimentversiontwelve/train/train_fixmatch.py --help
```

Tests use generated data and stored regression fixtures; the real EEG dataset
is not required. They cover PLV, dataset/graph behavior, original supervised
training results, FixMatch pseudo-label filtering, unlabeled data handling,
and fold integration. They do not establish full-dataset model performance.

If training reports missing files, set `--data-dir` and check the selected
subject filenames. If CUDA is unavailable, use `--device cpu` for FixMatch or
`device="cpu"` in the training configuration. If the pseudo-label acceptance
rate is zero in a direct engine call, no unlabeled graphs met the confidence
threshold; the update is then supervised only.
