smoke, full adn pooled will only run on train_test, train_valid_test, cross_validation_test and grid_search_test

loso will work only with loso yaml and inside has train, val, test logic

# BRAINDECODE

```bash
# Within-subject smoke run
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=bcic_iv_2a_within_subject_smoke

# Full within-subject run
python scripts/braindecode_scripts/train_within_subjects.py experiment=bcic_iv_2a_within_subject_full

# Subject-pooled run
python scripts/braindecode_scripts/train_subject_pooled.py experiment=bcic_iv_2a_subject_pooled

# Leave-one-subject-out run
python scripts/braindecode_scripts/train_loso.py experiment=bcic_iv_2a_loso
```

## Running multiple models / seeds in one command (Hydra multirun)

Model and seed sweeps are a **CLI thing**, not a config-file thing. Keep the
experiment yaml single-model (`override /model: eegnet`) and layer the sweep on
top with `-m` (multirun) plus a comma-list. Do NOT put the comma-list in the
yaml `defaults:` list — that selects one option and will error on a list.

```bash
# Same experiment, run once per model (sequential)
python scripts/braindecode_scripts/train_within_subject_smoke.py -m \
  experiment=bcic_iv_2a_within_subject_smoke model=eegnet,shallowfbcspnet,deep4net

# Sweep two axes at once -> every combination (here 3 models x 3 seeds = 9 runs)
python scripts/braindecode_scripts/train_within_subjects.py -m \
  experiment=bcic_iv_2a_within_subject_full model=eegnet,deep4net,atcnet seed=1,2,3
```

Notes:
- `model=a,b,c` (an override) sweeps; `override /model: a` (a defaults entry) does not.
- `train_within_subjects.py` and `train_loso.py` already loop subjects/folds
  internally, so a model sweep multiplies cleanly on top.
- The master CSV and TensorBoard still aggregate correctly under `outputs/`
  (their paths come from config, not Hydra's output dir). Only Hydra's per-run
  artifact folder (`model.pt`, `logs`) currently falls back to `multirun/`
  unless a `hydra.sweep.dir` is added to `configs/config.yaml`.

## Experiment tracking

Runs are organized for manuscript-friendly comparison across experiments, datasets, models, subjects/folds, and seeds.

```text
outputs/runs/{dataset}/{experiment}/{model}/{timestamp}__seed{seed}/
  logs/run.log
  metrics/final_metrics.yaml
  metrics/final_metrics.json
  metrics/dataset_info.yaml
  metrics/run_metadata.yaml
  history/history.csv
  checkpoints/model.pt
  results/

outputs/tensorboard/{dataset}/{experiment}/{model}/{run_id}/
outputs/results/{dataset}/results_master_{experiment}.csv
```

Use TensorBoard to compare all runs:

```bash
tensorboard --logdir outputs/tensorboard
```

Or compare one experiment/dataset across models:

```bash
tensorboard --logdir outputs/tensorboard/bcic_iv_2a/within_subject_smoke
```

Liu2024


```bash
# Within-subject smoke run
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=liu2024_within_subject_smoke

# Full within-subject run
python scripts/braindecode_scripts/train_within_subjects.py experiment=liu2024_within_subject_full

# # Subject-pooled run
python scripts/braindecode_scripts/train_subject_pooled.py experiment=liu2024_subject_pooled

# # Leave-one-subject-out run
python scripts/braindecode_scripts/train_loso.py experiment=liu2024_loso