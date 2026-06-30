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

`configs/config.yaml` is the top-level entry point with defaults `dataset: synthetic`, `preprocessing: motor_imagery`, `model: eegnet`, `training: default`, and `experiment: null`. An `experiment=<name>` preset under `configs/experiment/` is a `# @package _global_` override that swaps in dataset/preprocessing/training groups and sets `experiment_name`. Experiment presets are dataset-prefixed (`bcic_iv_2a_*`, `liu2024_*`).

Run output directories are templated in `config.yaml`:
```
outputs/runs/{dataset.label}/{experiment_name}/{model.label}/{timestamp}__seed{seed}/
```

Config groups:
- `dataset` — `synthetic`, `bcic_iv_2a`, `bcic_iv_2a_subject1`, `liu2024`, `liu2024_subject1`.
- `preprocessing` — `motor_imagery`, `bcic_iv_2a`, `liu2024`.
- `model` — `eegnet`, `shallowfbcspnet`, `deep4net`, `atcnet`, or any class in `braindecode.models` (resolved case-insensitively by name in `models/factory.py`, with `difflib` suggestions on typos).
- `training` — `default`: epochs, batch size, LR, `validation` block, `eval_metrics` list, `cross_validation`/`grid_search` knobs.

Note: `configs/experiment/bcic_iv_2a_within_subject_full.yaml` currently has uncommitted local edits (`session_grid_search_test` strategy, `max_epochs: 100`) — check `git diff` before assuming the committed defaults if behavior looks off.

### Split strategies (`src/eeg_bci/data/splitting/`)

This is a package, not a single file. Strategy name constants live in `splitting/strategies.py` and come in two parallel families that map to the same internal methodology via `STRATEGY_METHODS`/`split_method()`:
- `session_*` strategies (`session_train_test`, `session_train_valid_test`, `session_cross_validation_test`, `session_grid_search_test`, `session_leave_one_subject_out`) — used for BCI IV 2a, which has an official `session` column (`0train` for training, `1test` for held-out testing).
- `chronological_*` strategies (mirrored set, plus `chronological_train_test`/`chronological_leave_one_subject_out`) — used for datasets without a session column, e.g. Liu2024 (`dataset.split.strategy: chronological_train_test`), splitting by time order instead.
- `random` — used by the synthetic dataset.

`splitting/api.py` (`split_train_test`/`split_train_eval`) is a backward-compatible wrapper; `splitting/plans.py` (`make_braindecode_protocol_split`, `SplitPlan`) is the richer entry point used by `train_*` scripts for strategies that need a `SplitPlan` (train/valid/cross-val/grid-search/LOSO). `splitting/loso.py` builds one fold per held-out subject. `dataset.split.validation`/`dataset.split.resampling` configure inner validation/k-fold/holdout resampling *inside* the training pool — separate from `training.validation.enabled`, which carves a validation split out of the training set at the training-loop level.

### Data pipeline (`src/eeg_bci/data/`)

`datasets.py` (`build_dataset`, `build_dataset_split`) dispatches by `dataset.name`/`dataset.split.strategy` to `moabb.py` (loads MOABB datasets, sets local cache dir from `paths.py`, resolved relative to Hydra's original CWD so it survives Hydra's CWD change) or `synthetic.py`. `preprocessing.py` builds Braindecode `Preprocessor` pipelines from config; `windowing.py` turns continuous recordings into event windows and infers `DatasetInfo` (`n_chans`, `n_outputs`, `n_times`, `sfreq` — see `types.py`). `adapters.py` wraps Braindecode datasets as PyTorch `(x, y)` `Dataset`s.

### Training (`src/eeg_bci/braindecode_training/`)

`trainer.py` is the main dispatcher (`train_from_split_plan` etc.) — picks single-fit, `cross_val_score`, or `GridSearchCV` based on the split strategy's methodology (`TRAIN_TEST`, `TRAIN_VALID_TEST`, `CROSS_VALIDATION_TEST`, `GRID_SEARCH_TEST`, `LOSO`). `classifier.py` builds a Braindecode `EEGClassifier` (skorch) with AdamW + cosine LR schedule. `evaluation.py` scores classifiers and reads final-history metrics. `checkpointing.py` saves `state_dict` to `model.pt`.

### Tracking (`src/eeg_bci/tracking/`)

- `metrics.py` — `DEFAULT_METRICS`/`normalize_eval_metrics`, computes whichever metrics are requested via `training.eval_metrics` (accuracy, balanced_accuracy, cohen_kappa, macro_f1/precision/recall, confusion_matrix, roc_auc).
- `artifacts.py` — `prepare_run_dirs`, `save_final_metrics`, `save_dataset_info`, `save_run_metadata`, `export_history`.
- `tensorboard.py` — scalar/history logging, confusion-matrix heatmaps, run-text summaries.
- `results.py` — `MASTER_COLUMNS`, `master_result_path()`/`append_master_result()` for `outputs/results/<dataset>/results_master_<experiment>.csv`. Normalizes `test_acc` → `test_accuracy` for the master CSV.
- `naming.py` — run-id/label helpers (`dataset_label`, `model_label`, `subject_scope`, `tensorboard_dir`, `class_names_from_mapping`).
- `logging.py` — logging setup; MNE/Braindecode log levels are reduced to `WARNING`.

### Entry-point scripts (`scripts/braindecode_scripts/`)

Each `train_*.py` script: inserts `src/` onto `sys.path`, is a `@hydra.main` entry point reading `configs/config.yaml`, calls `seed_everything(cfg.seed)`, builds the dataset split, builds the model via `models/factory.py`, trains via `trainer.py`, then writes metrics/checkpoints through `tracking/`. Four scripts: `train_within_subject_smoke.py`, `train_within_subjects.py` (loops all subjects, per-subject checkpoints + `within_subject_results.csv`), `train_subject_pooled.py` (one shared model, optional `subject_pooled_test_results.csv` if multiple test subjects), `train_loso.py` (one fold per subject, `loso_results.csv`, per-fold checkpoints under `held_out_subject_<id>/`).

## Conventions

- Modern Python 3.11: `from __future__ import annotations`, type hints, dataclasses.
- Read config through `omegaconf.DictConfig`; convert with `OmegaConf.to_container(..., resolve=True)` when plain Python containers are needed.
- Use `pathlib.Path`; resolve dataset cache paths against Hydra's *original* CWD (see `data/paths.py`) since Hydra changes the working directory per run.
- Always call `seed_everything(cfg.seed)` (in `utils/seed.py`) at the start of a script; it seeds `random`, `numpy`, and `torch`/CUDA.
- Unsupported config values should raise `ValueError`/`TypeError` listing available options (see `models/factory.py`, `splitting/strategies.py` for the pattern, including `difflib` close-match suggestions where useful).

## Known sharp edges

- `training.eval_metrics` is configurable, but some trainer code paths assume `"accuracy"` is always present in the result — don't remove `accuracy` from a custom `eval_metrics` override without checking `trainer.py`.
- Confusion matrix / per-class metrics / ROC AUC in `metrics.py` are derived from `np.unique(y_true)` rather than a fixed label space, so folds/splits missing a class can produce mismatched matrix sizes or mislabeled per-class stats.
- LOSO summary aggregation in `train_loso.py` pairs metric values with fold counts positionally; a metric missing from one fold can silently misalign the weighted average.
