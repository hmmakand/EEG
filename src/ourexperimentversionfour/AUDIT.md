# Nested-LOSO protocol audit: experiment version four

Audit date: 2026-09-09

**Update (same date):** item 6's balanced accuracy, Cohen's kappa, and
confidence intervals have since been implemented (`training/metrics.py`,
`summarize_results` in `training/loso.py`) and are marked "Done" below.

**Update (same date, second pass):** item 2's inner cross-validation and
hyperparameter search, and item 4's retrain-with-selected-configuration, are
now implemented (`data/loso_split.py`'s `create_inner_cv_folds`,
`training/hyperparameter_search.py`, and `train_loso_fold_with_search`/
`train_all_loso_folds_with_search` in `training/loso.py`) and marked "Done"
below. Building this also surfaced and fixed an unrelated, pre-existing bug:
`training/engine.py`'s `set_seed()` never actually guaranteed
bit-reproducible training runs (PyTorch's default GCN scatter/gather kernels
aren't deterministic by default even with fixed seeds) -- see that module's
`set_seed()` docstring for the root-cause evidence. Item 3 (alignment,
feature selection, graph construction, augmentation) is still open.

**Update (2026-09-10):** unrelated logging bug fixed: `training/loso.py`'s
tqdm progress bars only redraw correctly on a real interactive terminal --
when stderr is piped/captured (not a tty), every animated-bar refresh left
stale bar text (e.g. `LOSO folds: 8%|...|`) glued onto the front of the next
per-epoch/per-fold log line, confirmed from a captured run where this
happened on every line. Fixed by adding `_bars_enabled()`, which disables
bar animation whenever `sys.stderr.isatty()` is false and falls back to the
existing plain `_log` lines (which already carry the same per-epoch/per-fold
information, so nothing is lost in non-interactive runs). Not a protocol-
compliance item -- noted here only because, like the `set_seed()` fix above,
it's a pre-existing bug this work surfaced, not new-feature scope.

## Requested protocol

1. Hold out one entire patient as the outer test fold.
2. Split the remaining 49 patients using grouped inner cross-validation for
   hyperparameters and early stopping.
3. Fit normalization, alignment, feature selection, graph construction and
   augmentation only inside the training portion.
4. Retrain on the 49 non-test patients with the selected configuration.
5. Evaluate the held-out patient exactly once.
6. Report all 50 patient-level results, mean +/- SD, balanced accuracy,
   kappa and confidence intervals.

## Scope and evidence

Covers `src/ourexperimentversionfour/data/`, `model/`, and `training/` as of
this audit. Verified by reading the executable code (not the README), and by
`grep -rniE "grid.?search|hyperparameter.*search|optuna|inner.*cv|nested.*cv|
k.?fold|kappa|balanced_accuracy|confidence.interval|augment"` across the
package, which returned zero matches -- confirming several gaps below by
absence, not just by inspection.

## Executive summary

Five of the six protocol items are now fully implemented: the outer LOSO
fold, inner cross-validation with hyperparameter search, retraining with the
selected configuration, the single held-out evaluation, and reporting. Only
item 3 remains open: its preprocessing list (alignment, feature selection,
graph construction, augmentation) is entirely absent except normalization,
which itself is scoped differently than requested (see below) -- it is fit
per-subject (including the test subject's own unlabeled data), not strictly
"inside the training portion only."

## Item-by-item findings

### 1. Hold out one entire patient as the outer test fold -- Implemented

`create_loso_splits` (`data/without_csd_alpha_wpli.py:214`) builds one fold
per subject present in the dataset, with `test_graph_indices` restricted to
that one subject. `train_all_loso_folds` (`training/loso.py:815`) iterates
every subject as the outer test fold and calls `train_loso_fold` once per
subject. Confirmed by a real 50-fold run's `loso_results.json` containing 50
distinct `test_subject_id` entries.

### 2. Grouped inner cross-validation for hyperparameters and early stopping -- Implemented

- **Grouped**: yes, both for the outer split and the new inner split.
  `_validate_split` (`data/loso_split.py`) checks the outer train/
  validation/test subject sets are pairwise disjoint; `_validate_inner_folds`
  (`data/loso_split.py`) checks the same for the inner folds, plus that every
  development subject appears in exactly one inner fold's validation set.
- **Cross-validation**: yes. `create_inner_cv_folds` (`data/loso_split.py`)
  splits each outer fold's 49 development subjects into `k=10` grouped
  folds via `sklearn.model_selection.GroupKFold(shuffle=True,
  random_state=seed)` -- 9 folds of 5 validation subjects (44 train) and 1
  fold of 4 (45 train), matching the requested 44-45/4-5 split exactly. Every
  development subject is used for validation exactly once per outer fold.
- **Hyperparameters**: searched via inner CV. `select_hyperparameters`
  (`training/hyperparameter_search.py`) scores a 12-candidate grid
  (`learning_rate` x `weight_decay`, `DEFAULT_SEARCH_GRID`) against the same
  10 inner folds for every candidate (so comparisons vary only the
  hyperparameter, not the data split), ranking candidates by mean validation
  **balanced accuracy** across folds -- chosen over loss because it is the
  field's own chance-corrected convention (Cohen's kappa is BCI Competition
  IV's official scoring metric for the same reason) and because item 6's
  diagnostics showed loss can look fine at chance-level discrimination.
  Within-run epoch selection inside each inner-CV training run still uses
  validation loss (unchanged, matching `train_loso_fold`'s existing
  criterion) -- balanced accuracy decides which *candidate* wins, not which
  *epoch* within a run. `batch_size`/`gradient_clip_norm` are valid
  `TrainingConfig` fields but not yet in the default grid (kept fixed to
  control candidate count); only `learning_rate`/`weight_decay` vary.
- **Early stopping**: yes, both within inner-CV runs (as above) and on the
  final retrain, via the pre-existing single grouped validation split
  (`_validate_normalization` at `training/loso.py`, and the epoch loop's
  early-stopping check).

### 3. Fit normalization, alignment, feature selection, graph construction, augmentation only inside the training portion

| Step | Status | Detail |
|---|---|---|
| Normalization | Implemented, but different scope | `fit_feature_normalization` (`data/without_csd_alpha_wpli.py:270`) computes one z-scoring mean/std **per subject**, from that subject's own raw trials -- for every subject in the dataset, including the held-out test subject and the validation subjects, not only the 44 training subjects. This is a deliberate choice (documented in the function's docstring and `training/README.md` as leakage-safe, since it never touches labels -- analogous to per-subject/session calibration such as Euclidean Alignment). It is nonetheless a literal deviation from "fit only inside the training portion": test-subject statistics come from that subject's own data, not from training-subject statistics applied to it. |
| Alignment | Not implemented | No covariance-based alignment (Euclidean Alignment, Riemannian Alignment, or similar) exists anywhere in the package. Only scalar per-column z-scoring of the 6 node features is performed. |
| Feature selection | Not implemented | Which node variant (`without_csd`/`csd`), edge variant (`wpli`), and band (`alpha`) to use is fixed by which combination module (`without_csd_alpha_wpli.py` / `csd_alpha_wpli.py`) is selected via config -- a hardcoded choice per combination, not a data-driven selection step fit on the training portion of each fold. |
| Graph construction | Not implemented as a fit step | `edge_index` (`_AlphaWpliGraphSubset.__init__`, `data/without_csd_alpha_wpli.py:100`) is copied directly from the saved dataset's fixed complete graph (all 406 undirected electrode pairs, `saved_dataset.py`), identical for every subject and every fold. Nothing about graph topology (thresholding, sparsification, learned adjacency) is fit from training data -- version three had a `plv_threshold`-based sparsification step, but that was removed when this package moved to a sparse `GCNConv` (see prior conversation); nothing replaced it. |
| Augmentation | Not implemented | No augmentation code (time-domain jitter, mixup, noise injection, etc.) exists in `data/` or `training/`. |

### 4. Retrain on the 49 non-test patients with the selected configuration -- Implemented

`train_loso_fold_with_search` (`training/loso.py`) implements exactly the
two-phase structure: (a) `select_hyperparameters` searches over the 49
development subjects via inner CV (item 2), producing a winning
`TrainingConfig` override; (b) it then calls the **existing, unmodified**
`train_loso_fold` with that selected config -- which trains on the 44
training subjects, uses the 5 validation subjects for early stopping, and
evaluates once on the held-out test subject, i.e. items 4 and 5 together,
reusing the already-verified retrain/evaluate path rather than
reimplementing it. `train_all_loso_folds_with_search` runs this for all 50
outer folds and records each fold's selected hyperparameters and full
candidate score table (`hyperparameter_search.json` per fold,
`selected_hyperparameters_by_subject.json` at the run level) for
provenance.

### 5. Evaluate the held-out patient exactly once -- Implemented

`test_metrics = evaluate(model, loaders.test, loss_function, device)`
(`training/loso.py:705`) is called exactly once, after the epoch loop ends
and the best validation checkpoint (`best_state`) has been reloaded into the
model. `loaders.test` is never touched anywhere else in the training loop --
only `loaders.train` and `loaders.validation` are used during the epoch
loop, so there is no test-set peeking before this single evaluation.

### 6. Report all 50 patient-level results, mean +/- SD, balanced accuracy, kappa, confidence intervals -- Implemented

| Item | Status | Detail |
|---|---|---|
| All 50 patient-level results | Implemented | `_save_loso_summary` writes `loso_results.json`/`.csv` with one entry per subject (`result.as_dict()` for all 50 `FoldResult`s), and `loso_results.csv` has one row per `test_subject_id`. |
| Mean +/- SD | Implemented | `summarize_results` (`training/loso.py`) computes `mean_{metric}`/`std_{metric}` across the 50 folds for loss, accuracy, balanced accuracy, F1, recall, precision, AUC, and kappa. |
| Balanced accuracy | Implemented | `ClassificationMetrics.balanced_accuracy` / `calculate_metrics` (`training/metrics.py`) computes it via `sklearn.metrics.balanced_accuracy_score` (mean per-class recall), alongside the existing plain accuracy and positive-class-only F1/recall/precision. |
| Cohen's kappa | Implemented | `ClassificationMetrics.cohens_kappa` / `calculate_metrics` (`training/metrics.py`) computes it via `sklearn.metrics.cohen_kappa_score`. |
| Confidence intervals | Implemented | `summarize_results` (`training/loso.py`) computes a 95% percentile-bootstrap CI (10,000 resamples by default, seeded for reproducibility) on the across-fold mean of every reported metric, stored as `ci95_low_{metric}`/`ci95_high_{metric}`. |

## Gap summary

| # | Requirement | Status |
|---|---|---|
| 1 | Outer LOSO test fold | Done |
| 2 | Grouped split (train/validation/test disjoint by subject) | Done |
| 2 | Inner **cross-validation** (k-fold/repeated, not one split) | Done |
| 2 | Hyperparameter search | Done |
| 3 | Normalization fit strictly on the training portion | Different scope (per-subject, incl. test) |
| 3 | Alignment | Missing |
| 3 | Feature selection | Missing |
| 3 | Graph construction fit on training data | Missing (graph is fixed a priori) |
| 3 | Augmentation | Missing |
| 4 | Retrain on the 49 with a selected configuration | Done |
| 5 | Evaluate the held-out patient exactly once | Done |
| 6 | All 50 patient-level results | Done |
| 6 | Mean +/- SD | Done |
| 6 | Balanced accuracy | Done |
| 6 | Cohen's kappa | Done |
| 6 | Confidence intervals | Done |

## Bottom line

The pipeline is now a genuine **nested**-CV LOSO protocol: for each of the 50
outer folds, hyperparameters are selected via 10-fold grouped inner
cross-validation over the 49 development subjects (ranked by mean validation
balanced accuracy), the winning configuration retrains on the 44 training
subjects with the pre-existing 5-subject grouped early-stopping split, and
the held-out test subject is evaluated exactly once -- aggregated with
mean/SD/95%-CI and chance-corrected metrics (balanced accuracy, kappa)
across all 50 subjects. Building this also surfaced and fixed a pre-existing,
unrelated reproducibility bug in `training/engine.py`'s `set_seed()` (see the
top-of-file update note). What remains open is item 3's preprocessing list:
none of alignment, feature selection, or graph construction are fit from
data (they are fixed choices), and normalization is fit per-subject rather
than strictly on the training portion (a deliberate, documented choice, not
an oversight). Closing that is new engineering work -- happy to scope it
next if useful.

Separately, a **within-subject classification diagnostic**
(`training/within_subject.py` / `within_subject_cli.py`) now exists
alongside this pipeline -- see `WITHIN_SUBJECT_PLAN.md` and the README's
"Within-subject classification (diagnostic)" section. It trains/evaluates on
one subject's own trials only, with no cross-subject generalization
involved, to isolate whether the LOSO chance-level result (above) is a
features/graph problem or specifically a cross-subject-generalization one.
This is a diagnostic tool, not a protocol-compliance item, so no table rows
above change.
