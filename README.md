# BRAINDECODE

EEG-BCI development workspace using Braindecode for datasets/models/training and Hydra for experiment configuration.

## First runnable training smoke test

```bash
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke
```

The smoke config uses BCI IV 2a subject 1 for a fast real-data pipeline check.


## Dataset cache

MOABB/MNE datasets are downloaded under `data/moabb` by default for project-local, predictable storage. The synthetic default dataset is generated in memory and is not saved to disk.

## Override examples

```bash
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke training.max_epochs=10 model.params.drop_prob=0.4
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke
python scripts/braindecode_scripts/train_within_subjects.py experiment=within_subject_full
```

Public datasets may download data through MOABB/MNE the first time they are used. BCI IV 2a uses Braindecode description-based splitting: `session=0train` for training and `session=1test` for final testing.

## Within-subject training and evaluation

Use `scripts/braindecode_scripts/train_within_subject_smoke.py` for the subject-1 smoke run and `scripts/braindecode_scripts/train_within_subjects.py` for within-subject training and evaluation. The full BCI IV 2a config uses subjects 1-9; the subject-1 experiment is kept as a faster real-data smoke test.

```bash
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke
python scripts/braindecode_scripts/train_within_subjects.py experiment=within_subject_full
```

Per-subject checkpoints are saved in per-subject output folders, and aggregate metrics are written to `within_subject_results.csv` inside the Hydra run directory.

## Split note

For BCI IV 2a, the default experiment follows Braindecode's session split: `0train` is used for training and `1test` is used for final testing. Optional validation is split only from the training set. Enable it with `training.validation.enabled=true`; tune against validation, then use `test_acc` as the final holdout score.
