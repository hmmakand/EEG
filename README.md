# BRAINDECODE

```bash
# Within-subject smoke run
python scripts/braindecode_scripts/train_within_subject_smoke.py experiment=within_subject_smoke

# Full within-subject run
python scripts/braindecode_scripts/train_within_subjects.py experiment=within_subject_full

# Subject-pooled run
python scripts/braindecode_scripts/train_subject_pooled.py experiment=subject_pooled

# Leave-one-subject-out run
python scripts/braindecode_scripts/train_loso.py experiment=loso
```

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

