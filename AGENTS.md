# AGENTS.md

> This file is written for AI coding agents that need to work in the BRAINDECODE repository. It summarizes the project structure, technology stack, build/test commands, conventions, and sharp edges discovered by inspecting the actual codebase.
>
> **Prior state:** before this write, the repository had no `AGENTS.md` file at the project root. The companion file `CLAUDE.md` contains additional context targeted at Claude Code.

## Project overview

BRAINDECODE is a Python research workspace for EEG motor-imagery decoding. It is packaged as `braindecode-eeg-dev` (importable under `eeg_bci`) and is built on top of:

- [Braindecode](https://braindecode.ai/) (`braindecode>=1.5`) for EEG models and training utilities.
- [MOABB](https://neurotechx.github.io/moabb/) (`moabb>=1.5`) for public EEG datasets.
- [MNE-Python](https://mne.tools/) (`mne>=1.12`) for EEG preprocessing.
- [PyTorch](https://pytorch.org/) (`torch>=2.0`) and [skorch](https://skorch.readthedocs.io/) for neural-network training.
- [Hydra](https://hydra.cc/) (`hydra-core>=1.3`) for config-driven experiments.

Two real datasets are wired up:

- **BCI Competition IV 2a** (`BNCI2014_001` via MOABB): 9 subjects, 4 classes (left hand, right hand, feet, tongue).
- **Liu2024** (`Liu2024` via MOABB): 50 subjects, 2 classes (left hand, right hand).

A **synthetic** dataset exists for fast smoke tests.

The workspace supports four evaluation modes:

1. `train_within_subject_smoke` – one subject, very short epochs.
2. `train_within_subjects` – one model per subject, then a summary CSV.
3. `train_subject_pooled` – one model trained on all configured subjects.
4. `train_loso` – canonical leave-one-subject-out (LOSO) cross-validation.

There is no production deployment, web service, or CI/CD pipeline; this is a local research/experimentation codebase.

## Key configuration files

| File | Purpose |
|------|---------|
| `pyproject.toml` | PEP 621 project metadata, dependencies, `src` package layout. |
| `requirements.txt` | Mostly commented manual-install notes; `pyproject.toml` is the source of truth. |
| `configs/config.yaml` | Top-level Hydra config with defaults for dataset, preprocessing, model, training, and experiment groups. |
| `configs/dataset/*.yaml` | Dataset definitions (`synthetic`, `bcic_iv_2a`, `liu2024`, single-subject variants). |
| `configs/preprocessing/*.yaml` | Preprocessing pipelines. |
| `configs/model/*.yaml` | Model presets (`eegnet`, `shallowfbcspnet`, `deep4net`, `atcnet`). |
| `configs/training/default.yaml` | Training hyperparameters, eval metrics, cross-validation and grid-search knobs. |
| `configs/experiment/*.yaml` | Experiment presets that override the default groups and set `experiment_name`. |
| `.devcontainer/devcontainer.json` | Dev-container configuration (Python 3.11, GPU-enabled). |

## Technology stack and runtime architecture

- **Language:** Python 3.11+ (project metadata says `requires-python = ">=3.11"`).
- **Build backend:** `setuptools.build_meta` with packages found under `src/`.
- **Virtual environment:** `.venv/` is present and committed-aware (listed in `.gitignore`).
- **Package import name:** `eeg_bci` (not `braindecode_eeg_dev`).
- **Hardware:** CUDA is optional. `device: auto` in config maps to CUDA when available, otherwise CPU.
- **Experiment runner:** Hydra changes the working directory per run into `outputs/runs/...`. Scripts resolve relative paths (e.g. `data/moabb`) against Hydra's original working directory.
- **Logging:** standard library `logging` with a file handler writing `logs/run.log` per run; Braindecode/MNE/skorch log levels are downgraded to `WARNING`.
- **Tracking:** TensorBoard logs under `outputs/tensorboard/`; master-result CSVs under `outputs/results/`.

## Build and install commands

All commands assume the repository root as the working directory.

```bash
# Create and activate the virtual environment (already present in this workspace)
python3.11 -m venv .venv
source .venv/bin/activate

# Editable install of the package (this is the normal development setup)
python -m pip install -e .

# Install with dev extras (pytest, jupyterlab, ipykernel)
python -m pip install -e ".[dev]"
```

If you ever need to refresh the runtime libraries manually, `requirements.txt` contains the rough install order, but prefer the declarative dependencies in `pyproject.toml`.

## Running experiments

All entry points live in `scripts/braindecode_scripts/` and are Hydra apps configured from `configs/config.yaml`. Each script inserts `src/` onto `sys.path`, so they work even without an editable install (though editable install is recommended).

```bash
# Smoke test on synthetic data (no download required)
python scripts/braindecode_scripts/train_within_subject_smoke.py

# BCI IV 2a
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=bcic_iv_2a_within_subject_smoke
python scripts/braindecode_scripts/train_within_subjects.py experiment=bcic_iv_2a_within_subject_full
python scripts/braindecode_scripts/train_subject_pooled.py experiment=bcic_iv_2a_subject_pooled
python scripts/braindecode_scripts/train_loso.py experiment=bcic_iv_2a_loso

# Liu2024
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=liu2024_within_subject_smoke
python scripts/braindecode_scripts/train_within_subjects.py experiment=liu2024_within_subject_full
python scripts/braindecode_scripts/train_subject_pooled.py experiment=liu2024_subject_pooled
python scripts/braindecode_scripts/train_loso.py experiment=liu2024_loso
```

Hydra overrides compose normally:

```bash
python scripts/braindecode_scripts/train_within_subject_smoke.py \
  experiment=bcic_iv_2a_within_subject_smoke \
  training.max_epochs=10 \
  model=eegnet \
  model.params.drop_prob=0.4
```

Compare runs with TensorBoard:

```bash
tensorboard --logdir outputs/tensorboard
```

## Test commands

```bash
source .venv/bin/activate
python -m pytest -q
```

As of the exploration, the suite reports **29 passed** with only a few third-party deprecation warnings. To run a single test file:

```bash
python -m pytest tests/test_metrics.py -q
```

`tests/conftest.py` adds `src/` to `sys.path` so tests can import `eeg_bci` directly.

## Code organization

```text
configs/              Hydra config groups and experiment presets
scripts/braindecode_scripts/
  train_within_subject_smoke.py   single-subject smoke runner
  train_within_subjects.py        per-subject within-subject runner
  train_subject_pooled.py         pooled-subject runner
  train_loso.py                   leave-one-subject-out runner
src/eeg_bci/
  data/                 dataset loading, preprocessing, windowing, splitting
    splitting/          source/method split architecture
  models/               model factory
  braindecode_training/ EEGClassifier construction, training dispatch, evaluation
  tracking/             metrics, artifacts, TensorBoard, results CSVs, naming
  utils/                seeding helper
```

### Module responsibilities

- `src/eeg_bci/data/datasets.py` – dispatches `build_dataset`/`build_dataset_split` by `dataset.name` (`moabb` or `synthetic`).
- `src/eeg_bci/data/moabb.py` – loads `MOABBDataset`, applies preprocessing, creates windows, returns `DatasetInfo`.
- `src/eeg_bci/data/synthetic.py` – generates a deterministic synthetic classification dataset. Note: its RNG is hardcoded to `np.random.default_rng(7)` and does **not** use `cfg.seed`.
- `src/eeg_bci/data/preprocessing.py` – builds Braindecode `Preprocessor` pipelines from Hydra config.
- `src/eeg_bci/data/windowing.py` – creates event windows and infers `n_chans`, `n_outputs`, `n_times`.
- `src/eeg_bci/data/adapters.py` – wraps Braindecode datasets as `(x, y)` PyTorch `Dataset`s.
- `src/eeg_bci/data/types.py` – `DatasetInfo` dataclass.
- `src/eeg_bci/data/splitting/` – the split architecture (see next section).
- `src/eeg_bci/models/factory.py` – resolves model names against `braindecode.models` (case-insensitive) and instantiates with `n_chans`, `n_outputs`, `n_times` plus any `model.params`.
- `src/eeg_bci/braindecode_training/trainer.py` – main dispatcher: single fit, cross-validation, grid search, LOSO.
- `src/eeg_bci/braindecode_training/classifier.py` – builds Braindecode `EEGClassifier` with AdamW + cosine LR schedule.
- `src/eeg_bci/braindecode_training/evaluation.py` – scoring helpers and metric collection.
- `src/eeg_bci/braindecode_training/checkpointing.py` – saves `model.pt` `state_dict`.
- `src/eeg_bci/tracking/metrics.py` – metric computation, normalization, and LOSO/within-subject summary aggregation.
- `src/eeg_bci/tracking/artifacts.py` – directory preparation, YAML/JSON/CSV artifact writing.
- `src/eeg_bci/tracking/results.py` – master-result CSV paths, column ordering, `append_master_result`.
- `src/eeg_bci/tracking/run_recording.py` – device resolution and shared master-row assembly.
- `src/eeg_bci/tracking/naming.py` – run/dataset/model/subject labels and TensorBoard directory layout.
- `src/eeg_bci/tracking/tensorboard.py` – scalar logging, confusion-matrix heatmaps, run text summaries.
- `src/eeg_bci/utils/seed.py` – `seed_everything()` for `random`, `numpy`, `torch` (and CUDA).

## Configuration system

`configs/config.yaml` is the Hydra entry point:

```yaml
defaults:
  - dataset: synthetic
  - preprocessing: motor_imagery
  - model: eegnet
  - training: default
  - _self_
  - experiment: null
```

Config groups:

- `dataset` – `synthetic`, `bcic_iv_2a`, `bcic_iv_2a_subject1`, `liu2024`, `liu2024_subject1`.
- `preprocessing` – `motor_imagery`, `bcic_iv_2a`, `liu2024`.
- `model` – `eegnet`, `shallowfbcspnet`, `deep4net`, `atcnet`. Any class in `braindecode.models` can be requested by name.
- `training` – `default`.
- `experiment` – dataset-prefixed presets (`bcic_iv_2a_*`, `liu2024_*`).

Experiment presets use `# @package _global_` and override groups plus `experiment_name`. They only change `dataset.split.method`, never `dataset.split.source`, because `source` is intrinsic to the dataset structure.

Run output directories are templated in `configs/config.yaml`:

```text
outputs/runs/{dataset.label}/{experiment_name}/{dataset.split.method}/{model.label}/{timestamp}__seed{seed}/
outputs/tensorboard/{dataset.label}/{experiment_name}/{dataset.split.method}/{model.label}/{run_name}/
outputs/results/{dataset.label}/{dataset.split.method}/results_master_{experiment_name}.csv
```

The `{dataset.split.method}` segment keeps artifacts from different split methods from mixing.

## Split architecture: source vs. method

The splitting code is under `src/eeg_bci/data/splitting/` and separates two orthogonal concepts:

- **`source`** – how `train_pool`/`test_set` are carved from the raw dataset. This is dataset-structure-dependent:
  - `session` – for BCI IV 2a's official `0train`/`1test` recording-protocol column.
  - `chronological` – for datasets without a session column (e.g. Liu2024); splits by recording-local time order.
  - `random` – for synthetic/smoke data.
- **`method`** – the dataset-agnostic training/evaluation methodology applied on top of `(train_pool, test_set)`:
  - `train_test`
  - `train_valid_test`
  - `cross_validation_test`
  - `grid_search_test`
  - `leave_one_subject_out`

A run's split label is `{source}_{method}` (e.g. `session_grid_search_test`).

To add a new dataset-structure-dependent source, write a `build_xxx_source` function and register it in `SOURCE_BUILDERS` (`src/eeg_bci/data/splitting/sources.py`). The rest of the pipeline (`plans.py`, `loso.py`, `trainer.py`) only deals with `method`.

### Subject-aware splitting

`SplitSource` carries an optional `groups` array: per-row subject IDs for `train_pool`. When populated (`session`, `chronological`), validation/resampling splits are computed **independently within each subject and then unioned** (`grouped_split_indices` in `src/eeg_bci/data/splitting/config.py`). This prevents a flat positional validation fold from landing entirely inside one subject when data is ordered subject-by-subject.

`PerGroupKFold` (`src/eeg_bci/data/splitting/resampling.py`) is the opposite of sklearn's `GroupKFold`: every subject is proportionally represented in every fold. `HoldoutSplit` supports the same grouped logic.

The chronological source also splits independently per subject (and per class when `stratify` is true) because `i_start_in_trial` is recording-local, not a global timeline.

## Data pipeline

1. `build_dataset(cfg.dataset, cfg.preprocessing)` loads a dataset (`moabb` or `synthetic`).
2. For MOABB:
   - `moabb.set_download_dir(str(resolve_data_dir(...)))` caches downloads under `data/moabb`.
   - `apply_preprocessing` runs the Braindecode preprocessor pipeline.
   - `create_event_windows` produces fixed-size windows.
   - `infer_window_info` returns `(n_chans, n_outputs, n_times)`; `n_outputs` is taken from `dataset.mapping` when present so the label space stays stable even if a subject/fold is missing a class.
3. `build_dataset_split` builds a `SplitPlan` via `make_protocol_split` (or LOSO folds via `make_leave_one_subject_out_folds`).
4. `TensorDatasetFromBraindecode` adapts Braindecode windows to `(x, y)` tensors.

## Training loop

`trainer.py` dispatches based on `split_plan.method`:

- `train_test`, `train_valid_test`, `LOSO` → `_train_once`.
- `cross_validation_test` → `_train_with_cross_validation` using `sklearn.cross_val_score`, then a final fit on the full training pool.
- `grid_search_test` → `_train_with_grid_search` using `sklearn.GridSearchCV`, then the best estimator.

All paths share `_persist_and_collect`, which evaluates on the held-out test set, checkpoints the model, exports history, writes per-subject results, and logs TensorBoard scalars.

`classifier.py` builds a Braindecode `EEGClassifier` (skorch wrapper) with:

- `CrossEntropyLoss`
- `AdamW` optimizer
- cosine annealing LR schedule
- `accuracy` callback
- optional TensorBoard callback
- `train_split=predefined_split(valid_set)` when a validation set exists

## Metrics

Default metrics (`src/eeg_bci/tracking/metrics.py`):

- `accuracy`
- `balanced_accuracy`
- `cohen_kappa`
- `macro_f1`, `macro_precision`, `macro_recall`
- `confusion_matrix`
- `roc_auc`

`accuracy` is always kept as an internal invariant because the trainer uses it as the canonical test score.

`compute_metrics` accepts an explicit `labels` array; the trainer passes `range(n_outputs)` derived from `dataset.mapping` so confusion matrices and ROC AUC stay stable when a split is missing classes.

`summarize_scalar_metrics` aggregates rows (e.g. per-subject/fold) with mean/std/min/max/weighted statistics, using `n_test_windows` as weights.

## Tracking and outputs

Each run produces:

```text
{output_dir}/
  logs/run.log
  metrics/final_metrics.yaml
  metrics/final_metrics.json
  metrics/dataset_info.yaml
  metrics/run_metadata.yaml
  history/history.csv
  checkpoints/model.pt
  results/                 # subject-level CSVs when applicable
```

`run_metadata.yaml` records Python/PyTorch/Braindecode versions, CUDA availability, git commit, and dirty-state.

Master-result CSVs live in `outputs/results/{dataset}/{method}/results_master_{experiment}.csv`. `results.py` defines `MASTER_COLUMNS` and `PREFERRED_METRIC_COLUMNS`; `order_fieldnames` is the single canonical column ordering reused by every per-run CSV.

## Code style guidelines

- Modern Python 3.11: `from __future__ import annotations`, type hints, dataclasses, `Protocol`.
- Use `pathlib.Path` for filesystem paths.
- Read config through `omegaconf.DictConfig`; convert to plain containers with `OmegaConf.to_container(..., resolve=True)` when needed.
- Resolve dataset/cache paths against Hydra's *original* CWD because Hydra changes the process directory per run.
- Always call `seed_everything(cfg.seed)` at the start of a script.
- Unsupported config values should raise `ValueError`/`TypeError` listing available options; use `difflib.get_close_matches` for helpful suggestions where appropriate.
- Do not keep backward-compatibility aliases for old split config keys. `dataset.split` must set both `source` and `method` explicitly (or be omitted entirely to default to `random`/`train_test`).
- There is no linter/formatter configured in this repo (no ruff/black/flake8 config) — do not invent one.

## Testing instructions

- Run the full suite: `source .venv/bin/activate && python -m pytest -q`.
- Run a single file: `python -m pytest tests/test_metrics.py -q`.
- The tests use lightweight fakes for Braindecode datasets and do not require downloading real EEG data.
- Key test files:
  - `tests/test_subject_aware_splitting.py` – grouped splits, `PerGroupKFold`, chronological groups.
  - `tests/test_chronological_split.py` – chronological split correctness.
  - `tests/test_loso_folds.py` – LOSO fold construction.
  - `tests/test_loso_summary.py` – weighted metric aggregation.
  - `tests/test_metrics.py` – metric computation and label-space stability.
  - `tests/test_naming.py` – TensorBoard/run directory naming.
  - `tests/test_results.py` – master CSV appending and legacy-key normalization.
  - `tests/test_run_recording.py` – master-row assembly.
  - `tests/test_windowing.py` – window-info inference.
  - `tests/test_tensorboard.py` – TensorBoard helpers.

## Deployment and operational notes

There is no deployment process. This is a local experimentation repository. Typical operational flow:

1. Edit Hydra configs or source code.
2. Run `pytest`.
3. Run one of the `train_*.py` scripts with an `experiment=...` preset.
4. Inspect `outputs/runs/` and `outputs/tensorboard/`.
5. Compare results in the master CSVs under `outputs/results/`.

The `.devcontainer/devcontainer.json` uses an official Python 3.11 image with `--gpus=all` and `--network=host`, so CUDA and dataset downloads work inside the container.

## Security considerations

- **No credentials or secrets** are stored in the repository. `.env` files are not used.
- **Public datasets only**: MOABB downloads public EEG datasets from standard repositories. Network access is required for first download; cached data lives in `data/moabb`.
- **Output data**: run artifacts under `outputs/` may contain model weights, metrics, and intermediate CSVs. These are local research artifacts and are gitignored by default.
- **TensorBoard**: `tensorboard --logdir outputs/tensorboard` opens a local network port. Only run it when you intend to expose logs on your local machine.
- **Virtual environment**: `.venv/` is local to the project. Do not install packages system-wide; use the venv.
- **No sandbox**: the workspace runs on the host filesystem. Be cautious when writing files outside the project root.

## Known sharp edges

- `training.eval_metrics` is configurable, but some trainer paths assume `"accuracy"` is present. Don't remove `accuracy` from a custom override without checking `trainer.py`.
- `compute_metrics` only falls back to observed labels when `labels=None`; the trainer always passes explicit labels, so direct callers of `compute_metrics` must supply `labels` if they need stability across missing classes.
- ROC AUC is rank-based and works with logits or probabilities. A split with only one class returns `NaN`. Aggregate ROC AUC in summaries is a per-subject/fold macro average (weighted by `n_test_windows`), not a single pooled-curve AUC.
- `synthetic.py` uses a hardcoded RNG (`np.random.default_rng(7)`) and does not respect `cfg.seed`.
- Chronological grouping is by `subject` only, not `(subject, session, run)`. This is safe for the currently wired datasets but would need extending for multi-session/multi-run datasets.
- LOSO is **canonical full-data LOSO**: each fold trains on *all* windows of the non-held-out subjects and tests on *all* windows of the held-out subject. The intra-subject `source` is only used for run labeling.

## Files to read first

If you are new to the codebase, read these in order:

1. `README.md` – quick command reference.
2. `CLAUDE.md` – detailed architecture notes from a prior audit.
3. `pyproject.toml` – dependencies and project metadata.
4. `configs/config.yaml` and `configs/experiment/*.yaml` – how experiments are composed.
5. `src/eeg_bci/data/datasets.py` and `src/eeg_bci/data/splitting/__init__.py` – the data/split entry points.
6. `src/eeg_bci/braindecode_training/trainer.py` – the training dispatcher.
7. `src/eeg_bci/tracking/results.py` and `src/eeg_bci/tracking/run_recording.py` – result aggregation.
