# Codex Review Report: Tracking and Metrics

Date: 2026-06-26

## Executive summary

The tracking and metrics work is a strong step forward. The project now records richer test metrics, writes structured artifacts, creates dataset/experiment/model-specific master result sheets, and logs TensorBoard summaries including confusion-matrix images. The generated within-subject results show that the new metrics are doing useful work: several EEGNet runs are near chance and the confusion matrices clearly expose single-class or two-class prediction collapse that plain accuracy alone would make too easy to miss.

The main remaining risks are around metric schema stability, fixed class-label handling, aggregation logic, and documentation drift. Most issues are not catastrophic for the current BCI IV 2a happy path because each subject has balanced four-class test data, but they will become real bugs for partial folds, synthetic/random splits, typoed configs, missing probabilities, future datasets, and manuscript-level aggregation.

Verification performed:

- Reviewed modified files in configs, scripts, `src/eeg_bci/braindecode_training/`, and `src/eeg_bci/tracking/`.
- Checked generated outputs under `outputs/runs/bcic_iv_2a/within_subject_full/...` and `outputs/results/bcic_iv_2a/results_master_within_subject_full.csv`.
- Ran `python -m compileall -q src scripts`; syntax compilation passed.
- Did not rerun full EEG training because it is data/GPU-expensive and the existing run outputs were available.

## What improved

1. Richer final metrics are now computed through `src/eeg_bci/tracking/metrics.py`.
   The default suite includes accuracy, balanced accuracy, Cohen kappa, macro F1, macro precision, macro recall, confusion matrix, and ROC AUC.

2. The training dispatcher now evaluates metrics consistently after single-fit, cross-validation, and grid-search final refits.
   Relevant entry points are in `src/eeg_bci/braindecode_training/trainer.py`.

3. TensorBoard support is better.
   `src/eeg_bci/tracking/tensorboard.py` writes scalar summaries and confusion-matrix heatmaps.

4. Result organization is more manuscript-friendly.
   The new master path format `outputs/results/<dataset>/results_master_<experiment>.csv` is much better than one global CSV.

5. The actual run outputs show useful diagnostic value.
   In `within_subject_results.csv`, EEGNet subject accuracies were roughly `0.2465` to `0.3090`, and several confusion matrices show almost all predictions going to one class. The later ShallowFBCSPNet rows in the master CSV are much stronger for several subjects, which suggests the new metrics are helping reveal model/config behavior rather than only logging decorative numbers.

## Priority findings

### 1. High: `eval_metrics` is configurable, but the trainer assumes `accuracy` is always present

Evidence:

- `configs/training/default.yaml:11-19` makes the metric list configurable.
- `src/eeg_bci/tracking/metrics.py:61-64` only computes metrics requested by the list.
- `src/eeg_bci/braindecode_training/trainer.py:186`, `285`, and `378` read `test_metrics["accuracy"]`.
- `src/eeg_bci/braindecode_training/evaluation.py:58` says the result always contains accuracy, but that is only true if the caller requested accuracy.

Impact:

If a user runs with an override such as `training.eval_metrics=[balanced_accuracy,cohen_kappa]`, training can finish and then crash during post-training evaluation with `KeyError: 'accuracy'`. This is especially painful because the failure happens after expensive model fitting.

Recommendation:

Make accuracy mandatory internally. Either always add `"accuracy"` to the requested set before evaluation, or compute `test_acc` through the legacy `classifier.score` path independently of the configurable report metrics. Also validate the config and raise a clear error before training if a required internal metric is missing.

### 2. High: confusion matrix, ROC AUC, and per-class metrics do not use a fixed label space

Evidence:

- `src/eeg_bci/tracking/metrics.py:89` calls `confusion_matrix(y_true, y_pred)` without explicit labels.
- `src/eeg_bci/tracking/metrics.py:99-116` uses `np.unique(y_true)` for per-class metrics.
- `src/eeg_bci/tracking/metrics.py:125-158` derives ROC AUC class count from `np.unique(y_true)`.
- `src/eeg_bci/tracking/tensorboard.py:103-126` assumes the matrix dimensions match `class_names`.

Impact:

This works for the current BCI IV 2a full test sessions because every subject appears to have all four classes. It can break or mislabel results when a test split, validation split, fold, synthetic split, or future dataset subset lacks one class. Specific failure modes:

- Confusion matrices become smaller than `n_outputs`.
- TensorBoard heatmap logging can index past the matrix size if `class_names` has four names but the matrix is `3x3`.
- Per-class values can be assigned to the wrong class names when a non-leading class is absent.
- Multiclass ROC AUC can be skipped because `y_prob.shape[1]` reflects the model's full class count while `np.unique(y_true)` reflects only classes present in that subset.

Recommendation:

Thread an explicit label list through evaluation, for example `labels=list(range(dataset_info.n_outputs))`. Use it in `confusion_matrix(..., labels=labels)`, per-class metrics, and ROC AUC. Also validate that `len(class_names) == len(labels)` before plotting.

### 3. High: LOSO weighted summaries can be numerically wrong when a metric is missing in any fold

Evidence:

- `scripts/braindecode_scripts/train_loso.py:184-205` summarizes every `test_*` key from the first result row.
- Values are filtered to rows where the metric exists, but `test_counts` still contains every fold.
- The weighted sum uses `zip(values, test_counts)`, which pairs a shortened metric list with the first N fold counts, not necessarily the matching fold counts.

Impact:

If a metric such as `test_roc_auc` is missing for one middle fold, the LOSO weighted average can silently use the wrong fold weights. Also, metrics absent from the first fold but present later are never summarized.

Recommendation:

Aggregate by row pairs, not by separate lists. For each metric, build `pairs = [(float(row[key]), int(row["n_test_windows"])) for row in results if key in row and is_number(row[key])]`, then compute mean/std/weighted from those pairs. Iterate over the union of result keys instead of `results[0]`.

### 4. Medium: within-subject parent runs do not save aggregate final metrics

Evidence:

- `scripts/braindecode_scripts/train_within_subjects.py:135-137` writes `within_subject_results.csv` and run metadata, but no parent-level `metrics/final_metrics.yaml`.
- LOSO does have parent-level summary saving in `scripts/braindecode_scripts/train_loso.py:169-172`.

Impact:

The within-subject experiment has per-subject final metrics, but the parent run has no mean/std/weighted summary artifact. That makes experiment comparison harder because `outputs/runs/.../metrics/final_metrics.yaml` is not available at the parent level for the most common full evaluation.

Recommendation:

Add a shared summarizer for grouped experiments and use it for within-subject and LOSO. Save at least mean, std, min, max, and window-weighted mean for scalar test metrics.

### 5. Medium: the master result CSV silently drops important new fields

Evidence:

- `src/eeg_bci/tracking/results.py:8-44` defines a fixed `MASTER_COLUMNS` list.
- `src/eeg_bci/tracking/results.py:59` uses `extrasaction="ignore"`.
- Current metrics may include `test_confusion_matrix`, `best_score`, `best_params`, fold scores, and `subject_test_results`, but many are not in `MASTER_COLUMNS`.

Impact:

The master CSV is useful, but it silently loses important traceability data. For grid search especially, the selected hyperparameters and best score are central to interpreting a run. For confusion matrices, dropping the nested matrix may be fine, but the master should at least link to the artifact path that contains it.

Recommendation:

Add columns for `best_score`, `best_params`, `best_score_std`, `best_train_score`, `best_train_score_std`, `subject_test_results`, and a path to final metrics. For complex metrics such as confusion matrices, store them in artifacts and put only artifact paths in the master sheet.

### 6. Medium: metric names are inconsistent across files and logs

Evidence:

- Trainer metrics include both `test_acc` and `test_accuracy` via `_prefix_test_metrics` in `src/eeg_bci/braindecode_training/trainer.py:443-445`.
- Master normalization maps `test_acc` to `test_accuracy` in `src/eeg_bci/tracking/results.py:68-76`.
- Per-subject pooled breakdown rows from `score_classifier_by_description` use unprefixed names like `accuracy`, not `test_acc` or `test_accuracy`.
- The generated `within_subject_results.csv` has both `test_acc` and `test_accuracy`.

Impact:

The same concept appears under multiple names, which increases the chance of downstream notebooks or manuscript tables using the wrong column. It also creates duplicate TensorBoard writes to the same `test/accuracy` tag.

Recommendation:

Choose one public schema. A good compromise is:

- Keep `test_acc` only as a backward-compatible alias inside Python if needed.
- Use `test_accuracy` in persisted artifacts.
- Use the same `test_*` prefix in grouped/per-subject CSVs.
- Avoid writing both `test_acc` and `test_accuracy` to TensorBoard.

### 7. Medium: the smoke-test preset is no longer clearly a smoke test

Evidence:

- `configs/experiment/within_subject_smoke.yaml:9-10` now sets `max_epochs: 10`.
- `configs/dataset/bcic_iv_2a_subject1.yaml:5-6` uses subject `3`, although the filename and project instructions describe subject 1.

Impact:

A smoke test should be fast and predictable. Ten epochs on real data is closer to a small experiment than a quick pipeline check. The subject mismatch also creates confusion when comparing logs and documentation.

Recommendation:

Decide the intent:

- If it is a true smoke test, use `max_epochs: 1` or `2`, restore subject `1`, and keep the run cheap.
- If subject `3` and ten epochs are intentional, rename the dataset config and update README/AGENTS text so the behavior is explicit.

### 8. Medium: unsupported metric names are silently ignored

Evidence:

- `src/eeg_bci/tracking/metrics.py:64` converts requested metrics to a set.
- Unknown names are not validated or reported.

Impact:

A typo like `macro_precisionn` produces a successful run with a missing metric. This is dangerous for long experiments because the mistake may only be noticed after the run is complete.

Recommendation:

Define `SUPPORTED_METRICS` and reject unknown names before training starts. If a metric cannot be computed at evaluation time, write `NaN` plus a reason field rather than silently omitting the key.

### 9. Low to medium: TensorBoard and README paths drifted

Evidence:

- Code now writes TensorBoard paths as `outputs/tensorboard/<dataset>/<experiment>/<model>/<run_id>` in `src/eeg_bci/tracking/naming.py:93-100`.
- README still documents `outputs/tensorboard/{experiment}/{dataset}/{model}/...` and `outputs/results_master.csv` in `README.md:21-33`.

Impact:

Users following the README will look in the wrong TensorBoard subtree and expect an old master result file.

Recommendation:

Update README paths to match the new dataset-first structure and grouped master files:

- `outputs/runs/{dataset}/{experiment}/{model}/{timestamp}__seed{seed}/`
- `outputs/tensorboard/{dataset}/{experiment}/{model}/{run_id}/`
- `outputs/results/{dataset}/results_master_{experiment}.csv`

### 10. Low: type annotations and schemas did not fully evolve with nested metrics

Evidence:

- `MetricValue = float | int | str` appears in `src/eeg_bci/braindecode_training/trainer.py` and `src/eeg_bci/tracking/artifacts.py`.
- New metrics can include nested lists (`test_confusion_matrix`) and potentially dicts (`per_class_recall`).
- Script result types such as `list[dict[str, str | float | int]]` do not match rows containing confusion matrices.

Impact:

Runtime currently works because the JSON/YAML helpers can serialize lists. But static tooling, future tests, and developer expectations will be misleading.

Recommendation:

Introduce a shared recursive JSON-like type, for example:

```python
MetricValue = str | int | float | bool | None | list["MetricValue"] | dict[str, "MetricValue"]
```

or use `Any` at persistence boundaries with explicit validation.

## Experiment-result observations

The opened EEGNet within-subject run shows near-chance behavior across all nine subjects. Several confusion matrices show one dominant predicted class, for example subject 3 predicts class 1 for every test window. Because BCI IV 2a has balanced test classes, `test_accuracy` and `test_balanced_accuracy` are identical in these rows. That equality is expected here and does not mean balanced accuracy is redundant for future datasets.

The master CSV also contains later ShallowFBCSPNet rows with substantially better subject-level results, for example subject 1 at about `0.5694` test accuracy and subject 3 at about `0.6458`. This supports the idea that the tracking changes are useful: they make model collapse and model-family differences visible immediately.

Recommended follow-up checks for model logic:

1. Compare EEGNet and ShallowFBCSPNet with the same epoch count and learning-rate schedule.
2. Confirm whether EEGNet needs model-specific parameters instead of relying on empty `params: {}`.
3. Add a short synthetic-data CI test so metric code is exercised without MOABB downloads.
4. Add a small deterministic test for a missing-class evaluation subset to protect confusion matrix and ROC AUC behavior.

## Suggested fix order

1. Make accuracy mandatory internally and validate metric names before training.
2. Thread explicit label IDs through `evaluate_classifier` and `compute_metrics`.
3. Fix LOSO aggregation to pair metric values with their own row counts.
4. Standardize persisted metric names and remove duplicate TensorBoard accuracy writes.
5. Add parent-level within-subject summary metrics.
6. Extend master CSV columns or add artifact path columns for fields currently dropped.
7. Update README and smoke-test naming/config drift.
8. Add focused pytest coverage for `compute_metrics`, LOSO summarization, master-row normalization, and TensorBoard confusion-matrix shape handling.

## Bottom line

The core direction is right. The new tracking layer is already surfacing valuable scientific signals, especially model collapse patterns. The main work left is to make the metric layer schema-stable, label-aware, and aggregation-safe so that future experiments cannot silently produce incomplete or misleading result tables.
