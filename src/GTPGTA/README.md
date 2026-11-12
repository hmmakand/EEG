# Version 29: validation and early stopping

An independent copy of version 27's model and dataset readers. Version 27 is unchanged.
All three datasets support LOSO and within-subject validation. No trained version 29
performance is claimed until the experiments are run.

## Defaults

- Maximum validation-stage training: **100 epochs** per model.
- Early stopping: **15 consecutive epochs** without a strict improvement in validation balanced accuracy.
- Earliest epoch wins ties (`min_delta=0`). The ensemble selects one shared epoch count.
- Three seed-ensemble members, Adam learning rate 0.001, batch size 32, and the version 27 architecture.
- Source-only calibration warm-up remains 10 epochs for LOSO. Selecting an earlier epoch is allowed;
  such a final refit ends before target-calibrated updates begin.
- The selection-stage models are discarded. Fresh models train on all non-test data for the selected
  number of epochs. There is no early stopping or epoch choice based on outer-test labels during refit.
- Test metrics are computed only at the final endpoint. Intermediate unlabeled target forwards may
  be necessary for calibration. No highest-test-epoch score is produced.

## LOSO split

Hold one subject out for the outer test. Select `ceil(0.2 * remaining_subjects)` validation subjects
with NumPy's RNG seeded by `[experiment_seed, held_out_subject]`; train on the rest.
Default cohorts and counts:

| Dataset | Training | Validation | Test |
|---|---:|---:|---:|
| Liu2024 | 36 | 9 | 1 |
| BCI2a | 6 | 2 | 1 |
| Jia2019 | 11 | 3 | 1 |

All members share these partitions. All trials from one subject stay together. Validation balanced
accuracy is averaged equally across validation subjects (not pooled across their samples).
The outer-test subject never enters the selection routine.

For calibrated LOSO, each validation subject has its own independent ensemble, source training data,
and pseudo-label state. A validation subject never calibrates another validation subject's model.
Validation target labels are used only to calculate validation metrics, never for adaptation.
This costs more than one ordinary training run: Liu2024 uses nine target-specific ensembles during
selection, then one fresh final ensemble. With calibration disabled, one source ensemble is shared
and each validation subject is scored/centered separately. Jia2019 has no electrode names and retains
version 27's disabled target calibration.

Subject-wise Euclidean alignment and ROI normalization retain version 27's transductive, unlabeled
subject-wise preparation. Each subject is prepared independently; the test subject cannot affect
training or validation preprocessing.

## Within-subject split

Preserve version 27's shuffled, trial-grouped outer 10 folds, seed 42. For zero-based outer test fold
`k`, use `(k + 1) % 10` for validation and the other eight for training. Trials are never split:
Liu2024 imagery/rest samples from one original trial stay together. Single-label trials are stratified.
At least three folds and both classes in every validation partition are required.

Select the epoch using the validation ensemble, then retrain from scratch on all nine non-test folds.
Every sample is tested once, held-out predictions are pooled within each subject, and the subject
metrics are averaged. No target calibration or logit centering is used within subject.

**Preprocessing change from version 27:** within-subject alignment is fitted only on selection-stage
training folds and frozen for validation. Final refit computes a new alignment from all non-test
folds, then applies that transform to the outer test fold. The test signal is not used to fit alignment.

**Jia2019 within-subject runs are supported but confounded:** the classes come from separate recordings.
A warning and `validity.valid_for_mi_comparison=false` are saved. Validation does not fix this confound.

## Run

From the repository root, with the project environment activated:

```bash
python -u -m src.GTPGTA.train.run_loso \
  --dataset liu2024 --data-dir /path/to/MNE-liu2024-data \
  --output-dir outputs/v29 --epochs 100 --patience 15 --device cuda --save-models

python -u -m src.GTPGTA.train.run_within \
  --dataset liu2024 --data-dir /path/to/MNE-liu2024-data \
  --output-dir outputs/v29 --epochs 100 --patience 15 --folds 10 --split-seed 42 \
  --device cuda --save-models
```

Use `--dataset bci2a` with the directory containing `A01T.mat` ... `A09T.mat`, or
`--dataset jia2019` with `exp1-S1-left.mat` ... . `--subjects 1` runs one outer subject;
LOSO still loads all source subjects. `--quiet` hides progress bars. `--epochs`, `--patience`,
`--min-delta`, `--seed`, and `--validation-fraction` (LOSO only) are configurable.
The existing `--paths` and `--no-target-calibration` (LOSO only) ablations remain available.

Run all datasets and both protocols sequentially:

```bash
bash src/GTPGTA/run_all.sh /path/to/MNE-liu2024-data /path/to/bci2a /path/to/jia2019 outputs/v29
```

`DEVICE=cpu` overrides the script's default CUDA device. The script saves final refit checkpoints.
These are full training experiments and can take substantially longer than version 27.

## Results

`<output-dir>/<dataset>/Version29_<dataset>_<task>_<protocol>_seed<seed>.json`, a readable summary,
and an embeddings NPZ are written. Existing result names receive a timestamp.
With `--save-models`, final refit checkpoints are stored below `models/<result stem>/` and contain
weights, architecture/data settings, the actual refit epoch count, and validation selection metadata.

Per-fold results include selection-stage subject or sample/trial assignments, validation balanced
accuracy history, best/stopping epochs, early-stopping reason, and final refit assignments.
Within-subject `test_index` specifies the prediction order. Training time includes both selection
and refit; graph construction is excluded. Model complexity remains single-model forward cost.
Version 27 analysis code expecting `highest_epoch` is not compatible with this intentional schema change.

```bash
python -m src.GTPGTA.analysis.figures /path/to/Version29_result.json
python -m unittest discover -s src/GTPGTA/tests -t .
```
