# AGENTS.md

This file documents the **BRAINDECODE** (package name `braindecode-eeg-dev`) project for AI coding agents. It is a Python research workspace for EEG-based brain–computer interface (BCI) decoding built on top of [Braindecode](https://braindecode.ai/), [MOABB](https://moabb.neurotechx.com/), [MNE-Python](https://mne.tools/), [PyTorch](https://pytorch.org/), and [Hydra](https://hydra.cc/).

---

## Project overview

BRAINDECODE trains deep-learning models on EEG motor-imagery data. The main supported dataset is **BCI Competition IV 2a** (exposed through MOABB as `BNCI2014_001`). A lightweight synthetic dataset is available for fast pipeline checks.

Key design decisions:

- **Configuration-driven experiments**: every run is controlled by Hydra YAML files under `configs/`.
- **Dataset/model/training decoupling**: the same training code works for synthetic data, single-subject smoke tests, full within-subject evaluations, subject-pooled training, and leave-one-subject-out cross-validation.
- **Session-aware splitting**: for real BCI IV 2a data, the official recording protocol is respected (`0train` for training, `1test` for final testing). Optional validation and inner cross-validation/grid-search are performed only inside the training pool.

---

## Technology stack

| Layer | Technology |
|-------|------------|
| Language | Python >= 3.11 |
| Deep learning | PyTorch >= 2.0 |
| EEG decoding | Braindecode >= 1.5 |
| Dataset access | MOABB >= 1.5, MNE >= 1.12 |
| Experiment config | Hydra >= 1.3 (OmegaConf) |
| Training wrapper | skorch (via `braindecode.EEGClassifier`) |
| Utilities | NumPy >= 2.0, pandas >= 3.0, scikit-learn >= 1.7, matplotlib >= 3.10, tensorboard >= 2.20 |
| Packaging | setuptools with `pyproject.toml` |

---

## Repository layout

```text
BRAINDECODE/
├── pyproject.toml              # setuptools project metadata
├── requirements.txt            # historical install notes (mostly commented out)
├── README.md                   # human-readable quick start
├── AGENTS.md                   # this file
├── .devcontainer/devcontainer.json
├── configs/                    # Hydra configuration hierarchy
│   ├── config.yaml             # top-level defaults
│   ├── dataset/
│   ├── experiment/
│   ├── model/
│   ├── preprocessing/
│   └── training/
├── data/moabb/                 # local MOABB/MNE cache (gitignored)
├── notebooks/                  # exploratory Jupyter notebooks
├── outputs/                    # Hydra run outputs (gitignored)
├── scripts/braindecode_scripts/# runnable entry points
│   ├── train_within_subject_smoke.py
│   ├── train_within_subjects.py
│   ├── train_subject_pooled.py
│   └── train_loso.py
└── src/eeg_bci/                # main package
    ├── data/                   # datasets, preprocessing, splitting, windowing
    ├── models/                 # Braindecode model factory
    ├── braindecode_training/   # skorch/Braindecode training loop
    └── utils/                  # seeding helpers
```

---

## Build and install

The project is packaged with `setuptools` and configured in `pyproject.toml`. The installable package lives under `src/`.

```bash
# Create and activate the virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install core dependencies
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126  # CUDA 12.6
python -m pip install -e .
```

> **Note:** the runnable scripts currently add `src/` to `sys.path` themselves, so an editable install is not strictly required to execute them. It is still recommended for imports and IDE support.

### Dev container

A VS Code dev container is configured in `.devcontainer/devcontainer.json` using `mcr.microsoft.com/devcontainers/python:3-3.11-trixie`.

---

## Running experiments

All experiments are launched through Hydra. The top-level config is `configs/config.yaml`.

### Smoke test (single subject, fast)

```bash
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke
```

This loads BCI IV 2a subject 1, applies dataset-specific preprocessing, trains `ShallowFBCSPNet` for 2 epochs, and reports `test_acc`. The script defaults to `experiment=within_subject_smoke` if no `experiment=` argument is supplied and refuses to run other experiment presets.

### Full within-subject evaluation

```bash
python scripts/braindecode_scripts/train_within_subjects.py experiment=within_subject_full
```

This loops over all configured subjects, trains one model per subject, and writes an aggregate `within_subject_results.csv` inside the Hydra run directory. Checkpoints are saved per subject under `outputs/<date>/<time>/subject_<id>/model.pt`.

### Subject-pooled evaluation

```bash
python scripts/braindecode_scripts/train_subject_pooled.py experiment=subject_pooled
```

This trains one shared model on all configured subjects' `0train` sessions and evaluates it on those same subjects' `1test` sessions. If the test set contains windows from more than one subject, per-subject accuracies are written to `subject_pooled_test_results.csv` in the run directory.

### Leave-one-subject-out evaluation

```bash
python scripts/braindecode_scripts/train_loso.py experiment=loso
```

This creates one fold per subject, training on all other subjects' `0train` sessions and testing on the held-out subject's `1test` session. Results are aggregated in `loso_results.csv`, and per-fold checkpoints are saved under `held_out_subject_<id>/model.pt`.

### Common overrides

```bash
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke training.max_epochs=10 model.params.drop_prob=0.4
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke model=eegnet
python scripts/braindecode_scripts/train_within_subjects.py experiment=within_subject_full training.max_epochs=20
python scripts/braindecode_scripts/train_subject_pooled.py experiment=subject_pooled training.max_epochs=20
python scripts/braindecode_scripts/train_loso.py experiment=loso training.max_epochs=20
```

---

## Configuration

Hydra config groups:

- `dataset` — `synthetic`, `bcic_iv_2a`, `bcic_iv_2a_subject1`.
- `preprocessing` — `motor_imagery`, `bcic_iv_2a` (currently identical values, separated so dataset-specific tuning can diverge).
- `model` — `eegnet`, `shallowfbcspnet`, `deep4net`, or any class exposed by `braindecode.models`.
- `training` — `default`: epochs, batch size, learning rate, validation/cross-validation/grid-search knobs, plus `eval_metrics` controlling which metrics `tracking/metrics.py` computes at test time.
- `experiment` — `within_subject_smoke`, `within_subject_full`, `subject_pooled`, `loso`: presets that override the groups above.

Key config conventions:

- `dataset.split.strategy` controls how data are split:
  - `random` — used by synthetic data.
  - `session_train_test`, `session_train_valid_test`, `session_cross_validation_test`, `session_grid_search_test` — used for real BCI IV 2a data.
  - `leave_one_subject_out` — used by the LOSO experiment.
- For session splits, the training pool is `session=0train` and the holdout test set is `session=1test`.
- `validation.enabled` in `training/default.yaml` adds a held-out validation split from the training set. For session splits, inner validation can also be configured via `dataset.split.validation` or `dataset.split.resampling`.

---

## Code organization

### `src/eeg_bci/data/`

| Module | Responsibility |
|--------|----------------|
| `datasets.py` | Top-level builder dispatch (`build_dataset`, `build_dataset_split`). |
| `moabb.py` | Load MOABB datasets (`BNCI2014_001`) and set the local download directory. |
| `synthetic.py` | Generate an in-memory synthetic dataset for smoke tests. |
| `preprocessing.py` | Build and apply Braindecode `Preprocessor` pipelines from config. |
| `windowing.py` | Convert continuous data into event windows and infer model input dimensions. |
| `splitting.py` | Split strategies: random, session-based, train/valid, cross-validation, grid-search resamplers, leave-one-subject-out folds. |
| `adapters.py` | Wrap Braindecode datasets into PyTorch `(x, y)` `Dataset`s. |
| `types.py` | `DatasetInfo` dataclass (`n_chans`, `n_outputs`, `n_times`, `sfreq`). |
| `paths.py` | Resolve dataset cache paths relative to Hydra's original working directory. |

### `src/eeg_bci/models/`

- `factory.py` builds any Braindecode model by name from `braindecode.models`. Model-specific hyperparameters come from `model.params` in config.

### `src/eeg_bci/braindecode_training/`

| Module | Responsibility |
|--------|----------------|
| `trainer.py` | Main training dispatcher: single fit, cross-validation, or grid search. |
| `classifier.py` | Construct a Braindecode `EEGClassifier` with AdamW, cosine LR schedule, and accuracy callbacks. |
| `evaluation.py` | Score classifiers and read final-history metrics. |
| `checkpointing.py` | Save the trained model `state_dict` as `model.pt`. |

### `src/eeg_bci/tracking/`

| Module | Responsibility |
|--------|----------------|
| `metrics.py` | Centralized metric computation (`accuracy`, `balanced_accuracy`, `cohen_kappa`, `macro_f1`, `confusion_matrix`, `roc_auc`, etc.). |
| `artifacts.py` | Save `final_metrics.yaml/json`, training history, dataset info, and run metadata. |
| `tensorboard.py` | Log scalars, run summaries, and confusion-matrix heatmaps to TensorBoard. |
| `results.py` | Define `MASTER_COLUMNS` and `master_result_path()` for grouped `outputs/results/<dataset>/results_master_<experiment>.csv` files. |
| `logging.py`, `naming.py` | Logging configuration and run-id/name helpers. |

### `src/eeg_bci/utils/`

- `seed.py` centralizes seeding for `random`, `numpy`, and `torch` (including CUDA).

---

## Development conventions

- **Python style**: modern Python 3.11 syntax, `from __future__ import annotations`, type hints, dataclasses, and docstrings.
- **Configuration values**: always read through `omegaconf.DictConfig`; convert to plain containers with `OmegaConf.to_container(..., resolve=True)` when needed.
- **Path handling**: use `pathlib.Path`. Dataset cache paths are resolved against Hydra's original CWD so relative paths survive Hydra's working-directory changes.
- **Randomness**: call `seed_everything(cfg.seed)` at the start of every script. Splits use seeded `torch.Generator` or `numpy.random.default_rng`.
- **Error messages**: unsupported values raise `ValueError`/`TypeError` with lists of available options.
- **Logging/tracking**: use `src/eeg_bci/tracking/` for Python logging, artifact saving, run naming, TensorBoard text/scalars, and `outputs/results_master.csv`. MNE/Braindecode log levels are reduced to `WARNING`.

---

## Testing

There is currently **no automated test suite** (no `tests/` directory, no `pytest` configuration, no CI/CD pipeline). Validation is done by running the smoke experiment:

```bash
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke
```

If you add tests, use `pytest` and place them in a top-level `tests/` directory. Ensure any new tests mock or skip large MOABB downloads.

---

## Outputs and artifacts

Hydra creates organized run directories under `outputs/runs/{dataset}/{experiment}/{model}/{timestamp}__seed{seed}/` for every invocation:

- `model.pt` — saved `state_dict` of the trained network.
- `.hydra/` — resolved config and overrides.
- `train_within_subject_smoke.log` / `train.log` / `train_loso.log` — captured stdout/log.
- `metrics/final_metrics.yaml` — final metrics, including `test_acc`, `test_balanced_accuracy`, `test_cohen_kappa`, `test_macro_f1`, `test_macro_precision`, `test_macro_recall`, `test_roc_auc`, and `test_confusion_matrix`.
- TensorBoard logs now include a `test/confusion_matrix` heatmap image when `confusion_matrix` is in `eval_metrics`.
- `outputs/results/<dataset>/results_master_<experiment>.csv` — per-dataset, per-experiment master result sheet (e.g., `outputs/results/bcic_iv_2a/results_master_within_subject_full.csv`).
- For within-subject runs: `subject_<id>/model.pt` and `within_subject_results.csv`.
- For subject-pooled runs: `subject_pooled_test_results.csv` when multiple subjects are in the test set.
- For LOSO runs: `held_out_subject_<id>/model.pt`, `loso_results.csv`, and `metrics/final_metrics.yaml` with per-fold summaries (`loso_accuracy_mean`, `loso_cohen_kappa_mean`, etc.).

These directories are gitignored by `.gitignore`.

---

## Security considerations

- No secrets, credentials, or API keys are stored in the repository.
- `.gitignore` excludes virtual environments (`.venv/`), editor settings (`.vscode/`, `.idea/`), environment files (`.env`), and large EEG data artifacts.
- MOABB datasets are downloaded from public sources and cached locally in `data/moabb/` (gitignored).
- Do not commit model checkpoints (`.pt`, `.pth`), raw EEG files (`.edf`, `.fif`, etc.), or Hydra output directories.

---

## Quick command reference

```bash
# Editable install
python -m pip install -e .

# Fast real-data smoke test
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke

# Full within-subject evaluation
python scripts/braindecode_scripts/train_within_subjects.py experiment=within_subject_full

# Subject-pooled evaluation
python scripts/braindecode_scripts/train_subject_pooled.py experiment=subject_pooled

# Leave-one-subject-out evaluation
python scripts/braindecode_scripts/train_loso.py experiment=loso

# Override training epochs and model dropout
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke training.max_epochs=10 model.params.drop_prob=0.4

# TensorBoard (if logs are generated)
tensorboard --logdir outputs/
```
