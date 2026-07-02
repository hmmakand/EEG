# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

EEG motor-imagery decoding research workspace (package `braindecode-eeg-dev`, importable as `eeg_bci`), built on Braindecode, MOABB, MNE-Python, PyTorch, skorch, and Hydra. Two datasets are wired up: **BCI Competition IV 2a** (`BNCI2014_001` via MOABB, 9 subjects, 4 classes) and **Liu2024** (`Liu2024` via MOABB, 50 subjects). Every run is config-driven through Hydra; the same training code supports synthetic smoke data, single-subject runs, full within-subject evaluation, subject-pooled training, and leave-one-subject-out (LOSO) cross-validation.

## Commands

```bash
# Install (editable; src/ layout, scripts also self-insert src/ onto sys.path)
python -m pip install -e .
python -m pip install -e ".[dev]"   # adds pytest, jupyterlab, ipykernel

# Run tests
pytest
pytest tests/test_metrics.py -k some_test   # single test

# BCI IV 2a experiments
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=bcic_iv_2a_within_subject_smoke
python scripts/braindecode_scripts/train_within_subjects.py experiment=bcic_iv_2a_within_subject_full
python scripts/braindecode_scripts/train_subject_pooled.py experiment=bcic_iv_2a_subject_pooled
python scripts/braindecode_scripts/train_loso.py experiment=bcic_iv_2a_loso

# Liu2024 experiments (same scripts, different experiment preset)
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=liu2024_within_subject_smoke
python scripts/braindecode_scripts/train_within_subjects.py experiment=liu2024_within_subject_full
python scripts/braindecode_scripts/train_subject_pooled.py experiment=liu2024_subject_pooled
python scripts/braindecode_scripts/train_loso.py experiment=liu2024_loso

# Hydra overrides compose normally, e.g.
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=bcic_iv_2a_within_subject_smoke training.max_epochs=10 model=eegnet model.params.drop_prob=0.4

# Compare runs in TensorBoard
tensorboard --logdir outputs/tensorboard
tensorboard --logdir outputs/tensorboard/bcic_iv_2a/bcic_iv_2a_within_subject_smoke
```

There is no lint/format command configured in this repo (no ruff/black/flake8 config found) — don't invent one.

## Architecture

### Config composition (Hydra)

`configs/config.yaml` is the top-level entry point with defaults `dataset: synthetic`, `preprocessing: motor_imagery`, `model: eegnet`, `training: default`, and `experiment: null`. An `experiment=<name>` preset under `configs/experiment/` is a `# @package _global_` override that swaps in dataset/preprocessing/training groups and sets `experiment_name`. Experiment presets are dataset-prefixed (`bcic_iv_2a_*`, `liu2024_*`) and only ever override `dataset.split.method` (and method-specific knobs), never `dataset.split.source` — `source` is intrinsic to the dataset, not the experiment.

Run output directories are templated in `config.yaml`, with `dataset.split.method` as its own path segment so changing the method between runs of the same experiment preset never mixes artifacts:
```
outputs/runs/{dataset.label}/{experiment_name}/{dataset.split.method}/{model.label}/{timestamp}__seed{seed}/
outputs/tensorboard/{dataset.label}/{experiment_name}/{dataset.split.method}/{model.label}/{run_name}/
outputs/results/{dataset.label}/{dataset.split.method}/results_master_{experiment_name}.csv
```

Config groups:
- `dataset` — `synthetic`, `bcic_iv_2a`, `bcic_iv_2a_subject1`, `liu2024`, `liu2024_subject1`. Each sets `dataset.split.source` (dataset-structure-dependent) and a default `dataset.split.method`.
- `preprocessing` — `motor_imagery`, `bcic_iv_2a`, `liu2024`.
- `model` — `eegnet`, `shallowfbcspnet`, `deep4net`, `atcnet`, or any class in `braindecode.models` (resolved case-insensitively by name in `models/factory.py`, with `difflib` suggestions on typos).
- `training` — `default`: epochs, batch size, LR, `validation` block, `eval_metrics` list, `cross_validation`/`grid_search` knobs.

### Split source/method (`src/eeg_bci/data/splitting/`)

This is a package, not a single file. A dataset's `split` config sets two independent, orthogonal keys (`splitting/strategies.py`):
- **`source`** — how `train_pool`/`test_set` are carved out of the raw dataset. Dataset-structure-dependent: `session` (BCI IV 2a's official `0train`/`1test` recording-protocol column), `chronological` (datasets without a session column, e.g. Liu2024 — splits by time order instead), `random` (synthetic data).
- **`method`** — the dataset-agnostic training/evaluation methodology applied on top of the resulting `(train_pool, test_set)` pair: `train_test`, `train_valid_test`, `cross_validation_test`, `grid_search_test`, `leave_one_subject_out`.

A run's full split label is `f"{source}_{method}"` (`split_label()`), e.g. `session_grid_search_test`.

Adding a new dataset-structure-dependent source only requires writing one `build_xxx_source(dataset, split_cfg, *, seed) -> SplitSource` function and registering it in `SOURCE_BUILDERS` (`splitting/sources.py`) — no changes needed in `plans.py`, `loso.py`, or `trainer.py`, since those only ever deal with `method`.

Key modules:
- `strategies.py` — the `SOURCE_*`/`*_TEST`/`LOSO` string constants, `METHODS` set, `split_label()`.
- `config.py` — `resolved_source_and_method()` (reads `split.source`/`split.method`, raises if either is missing), `resolve_subject_column()`, `grouped_split_indices()` (see "Subject-aware splitting" below).
- `sources.py` — `SOURCE_BUILDERS` registry (`session` → `make_session_split_source`, `chronological` → `make_chronological_split_source`, `random` → `build_random_source`); `build_split_source(source, dataset, split_cfg, *, seed)` is the dispatch entry point. `SourceBuilder` is a `Protocol` (not a `Callable` alias) so it can express the keyword-only `seed` param.
- `plans.py` — `make_protocol_split()` is the single entry point used by `train_*` scripts to get a `SplitPlan` for any non-LOSO method (train/valid/cross-val/grid-search). `SplitPlan` carries `train_pool`, `train_set`, `valid_set`, `test_set`, `resampler`, `method`.
- `loso.py` — `make_leave_one_subject_out_folds()` builds one `CrossSubjectFold` per held-out subject. This is **full-data (canonical) LOSO**: each fold trains on *all* windows of the non-held-out subjects (their whole per-subject dataset from `dataset.split("subject")`, wrapped via `TensorDatasetFromBraindecode`) and tests on *all* windows of the held-out subject. The intra-subject split `source` does **not** subset a subject inside a fold — it is read only to validate the method and to label the run via `split_label(source, LOSO)` (so output paths/labels are unchanged). LOSO therefore does not call `build_split_source`.
- `resampling.py` — `make_resampler()`/`make_chronological_resampler()` build the inner validation/k-fold resampler from `dataset.split.resampling`.

`dataset.split.validation`/`dataset.split.resampling` configure inner validation/k-fold/holdout resampling *inside* the training pool — separate from `training.validation.enabled`, which carves a validation split out of the training set at the training-loop level.

#### Subject-aware splitting (`SplitSource.groups`)

Every `SplitSource` (returned by a source builder) carries an optional `groups: np.ndarray | None` — the per-row subject id for `train_pool`. When a source can populate it (`session`, `chronological`), it flows through `make_split_plan`/`make_train_valid_test_plan`/`make_resampled_plan`/`make_leave_one_subject_out_folds` into:
- `split_train_valid(..., groups=...)` (`plans.py`)
- `HoldoutSplit(..., groups=...)` and `PerGroupKFold(..., groups=...)` (`resampling.py`)

all of which compute the split/fold **independently within each group and union the result**, via `grouped_split_indices()` (`config.py`), instead of taking a flat positional (or globally-shuffled) cut. This matters because pooled multi-subject `train_pool`s are ordered subject-by-subject — a plain positional cut (or sklearn's default `KFold(shuffle=False)`) can land an entire validation set or CV fold inside one or two subjects, silently skewing model selection. `PerGroupKFold` is the *opposite* of sklearn's `GroupKFold` (which keeps a whole group within one fold, for LOSO-style isolation) — here every group is proportionally represented in every fold/holdout, never isolated.

Shuffling, when enabled, only ever happens *within* a single group's own rows — groups (subjects) are never mixed together, preserving Braindecode's "never shuffle across time-correlated samples" guidance for time-series EEG data (see the [train/test/tune tutorial](https://braindecode.org/dev/auto_examples/model_building/plot_how_train_test_and_tune.html)).

The chronological source also fixed an analogous bug at the outer train/test cut: `i_start_in_trial` is a sample offset relative to *each subject's own recording*, not a timeline shared across subjects, so `_chronological_split_indices()` (`sources.py`) computes the chronological cutoff independently per subject (and per class, when `stratify`) before unioning — never sorting/cutting pooled subjects on a global `i_start_in_trial` ordering. Regression tests: `tests/test_chronological_split.py`, `tests/test_subject_aware_splitting.py`.

Known remaining latent assumption: chronological grouping is by `subject` only, not `(subject, session, run)` — currently safe since both wired-up datasets have exactly one session × one run per subject, but would need extending if that ever changes.

### Data pipeline (`src/eeg_bci/data/`)

`datasets.py` (`build_dataset`, `build_dataset_split`) dispatches by `dataset.name` to `moabb.py` (loads MOABB datasets, sets local cache dir from `paths.py`, resolved relative to Hydra's original CWD so it survives Hydra's CWD change) or `synthetic.py`, then calls `make_protocol_split()` to resolve `dataset.split.source`/`method` into a `SplitPlan`. `preprocessing.py` builds Braindecode `Preprocessor` pipelines from config; `windowing.py` turns continuous recordings into event windows and infers `DatasetInfo` (`n_chans`, `n_outputs`, `n_times`, `sfreq` — see `types.py`). `adapters.py` wraps Braindecode datasets as PyTorch `(x, y)` `Dataset`s.

`windowing.py:infer_window_info(windows, *, n_outputs=None)` now takes an explicit `n_outputs`; `moabb.py` passes `n_outputs_from_mapping(dataset.mapping)` (= `max(target)+1`) so the model head and metric label space stay stable even when a subject/fold is missing a class. It falls back to observed unique targets only when no mapping is configured (e.g. `synthetic`, which sets `n_outputs` from its own config). Remaining low-priority latent issue: `synthetic.py` uses a hardcoded `np.random.default_rng(7)` rather than deriving from `cfg.seed`.

### Training (`src/eeg_bci/braindecode_training/`)

`trainer.py` is the main dispatcher (`train_from_split_plan` etc.) — picks single-fit, `cross_val_score`, or `GridSearchCV` based on `split_plan.method` (`TRAIN_TEST`, `TRAIN_VALID_TEST`, `CROSS_VALIDATION_TEST`, `GRID_SEARCH_TEST`, `LOSO`). `classifier.py` builds a Braindecode `EEGClassifier` (skorch) with AdamW + cosine LR schedule. `evaluation.py` scores classifiers and reads final-history metrics. `checkpointing.py` saves `state_dict` to `model.pt`.

### Tracking (`src/eeg_bci/tracking/`)

- `metrics.py` — `DEFAULT_METRICS`/`normalize_eval_metrics`, computes whichever metrics are requested via `training.eval_metrics` (accuracy, balanced_accuracy, cohen_kappa, macro_f1/precision/recall, confusion_matrix, roc_auc).
- `artifacts.py` — `prepare_run_dirs`, `save_final_metrics`, `save_dataset_info`, `save_run_metadata`, `export_history`.
- `tensorboard.py` — scalar/history logging, confusion-matrix heatmaps, run-text summaries.
- `results.py` — `MASTER_COLUMNS`, `master_result_path(original_cwd, experiment, dataset, method)` → `outputs/results/{dataset}/{method}/results_master_{experiment}.csv` (the `{method}` directory segment keeps rows from different `dataset.split.method` runs of the same experiment preset from mixing into one CSV). `append_master_result()` normalizes `test_acc` → `test_accuracy` etc. for the master CSV. `PREFERRED_METRIC_COLUMNS` + `order_fieldnames(rows, leading=...)` is the single canonical CSV column ordering reused by every per-run CSV (scripts and `trainer._dynamic_fieldnames`) — don't reintroduce per-file preferred-column lists.
- `run_recording.py` — `resolve_device(name)` and `build_master_row(cfg, metrics, dataset_info, *, run_id, run_dir, hydra_output_dir, tb_dir, timestamp, subject=None, held_out_subject=None)`: the shared master-CSV row assembly and device resolution used by all four `train_*` scripts (`run_dir` anchors artifact paths, `hydra_output_dir` anchors the resolved-config path; they're identical for single-run scripts).
- `naming.py` — run-id/label helpers (`dataset_label`, `model_label`, `subject_scope`, `experiment_label` — falls back to `resolved_split_label(cfg.dataset.split)` when `experiment_name` isn't set —, `tensorboard_dir` which inserts `cfg.dataset.split.method` between the experiment and model segments, `class_names_from_mapping`).
- `logging.py` — logging setup; MNE/Braindecode log levels are reduced to `WARNING`.

### Entry-point scripts (`scripts/braindecode_scripts/`)

Each `train_*.py` script: inserts `src/` onto `sys.path`, is a `@hydra.main` entry point reading `configs/config.yaml`, calls `seed_everything(cfg.seed)`, builds the dataset split, builds the model via `models/factory.py`, trains via `trainer.py`, then writes metrics/checkpoints through `tracking/` (device via `run_recording.resolve_device`, master row via `run_recording.build_master_row`, per-run CSV columns via `results.order_fieldnames`), passing `str(cfg.dataset.split.method)` into `master_result_path()`. Four scripts: `train_within_subject_smoke.py`, `train_within_subjects.py` (loops all subjects, per-subject checkpoints + `within_subject_results.csv`), `train_subject_pooled.py` (one shared model, optional `subject_pooled_test_results.csv` if multiple test subjects), `train_loso.py` (one full-data fold per subject, `loso_results.csv`, per-fold checkpoints under `held_out_subject_<id>/`).

`trainer.py`'s three training paths (`_train_once`, `_train_with_cross_validation`, `_train_with_grid_search`) share a single tail, `_persist_and_collect()`, which evaluates the classifier, checkpoints, exports history, writes subject results, assembles the base metrics dict, and logs/writes TensorBoard scalars — each path only runs its own fit and supplies its path-specific `extra` metrics (`valid_*`, `cv_*`, `best_*`).

## Conventions

- Modern Python 3.11: `from __future__ import annotations`, type hints, dataclasses.
- Read config through `omegaconf.DictConfig`; convert with `OmegaConf.to_container(..., resolve=True)` when plain Python containers are needed.
- Use `pathlib.Path`; resolve dataset cache paths against Hydra's *original* CWD (see `data/paths.py`) since Hydra changes the working directory per run.
- Always call `seed_everything(cfg.seed)` (in `utils/seed.py`) at the start of a script; it seeds `random`, `numpy`, and `torch`/CUDA.
- Unsupported config values should raise `ValueError`/`TypeError` listing available options (see `models/factory.py`, `splitting/strategies.py` for the pattern, including `difflib` close-match suggestions where useful).
- No backward-compatibility layers are kept for the split config: `dataset.split` must always set both `source` and `method` explicitly (or be omitted entirely, which defaults to `random`/`train_test`). There is exactly one active codebase shape at a time — don't reintroduce aliasing for old config keys.

## Known sharp edges

- `training.eval_metrics` is configurable, but some trainer code paths assume `"accuracy"` is always present in the result — don't remove `accuracy` from a custom `eval_metrics` override without checking `trainer.py`.
- `compute_metrics()` in `metrics.py` falls back to `np.unique(y_true ∪ y_pred)` for its label space **only when `labels=None`**. In practice the trainer always passes `labels=range(n_outputs)` (from `dataset.mapping`, see Phase-2 note above), so confusion matrices / per-class metrics / ROC AUC are stable even for folds/splits missing a class — the fallback only bites *direct* callers of `compute_metrics` that omit `labels`.
- ROC AUC (`metrics.py`) is computed correctly: binary uses the positive-class probability column; multiclass uses one-vs-rest macro averaging with columns aligned to `labels`. It is rank-based, so whether the model emits logits or (log-)softmax is irrelevant to the value. A split with only one class present yields `NaN` (caught, logged). Aggregate ROC AUC in the summaries is a **per-subject/-fold macro average** (weighted by `n_test_windows` in the `_weighted` variant) with `NaN` folds dropped — report it as such, not as a single pooled-curve AUC.
- LOSO/within-subject summary aggregation (`summarize_scalar_metrics` → `_numeric_weight_pairs`) reads each metric value and its `n_test_windows` weight from the **same row** and skips a row entirely if either is missing — so there is no positional misalignment (this was a former sharp edge, now safe).
- Chronological grouping is by `subject` only, not `(subject, session, run)` (see "Subject-aware splitting" above) — fine today, but a latent assumption if a dataset with multiple sessions/runs per subject is added without a session-based source.
- `synthetic.py`'s RNG seed is hardcoded (`np.random.default_rng(7)`) rather than derived from `cfg.seed`.
