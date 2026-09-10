# Within-subject classification (diagnostic sanity check)

Status: **planned, not yet implemented** -- saved for review before execution.

## Context

The nested-CV LOSO pipeline (`training/loso.py`, `training/search_train.py`)
is complete and audited (`AUDIT.md`), but every run -- fixed hyperparameters
or searched -- sits at chance level (~50% balanced accuracy). Digging into
the hyperparameter-search results (subjects 1-8 of `csd_search_run_25`)
showed the 12-candidate grid's best-vs-worst spread is only ~1-3 points of
balanced accuracy, no candidate wins consistently, and the searched
hyperparameters don't even beat the plain default on held-out test -- i.e.
hyperparameter tuning has nothing to find, which points at something
upstream of optimization (features, graph, or genuine task difficulty)
rather than undertuned training.

The agreed next step is a **within-subject classification** experiment:
train and evaluate on one subject's own trials only, with no cross-subject
generalization involved at all. This isolates two separate questions the
LOSO number conflates:

1. **Can the model fit this data at all?** (an implicit "overfit a small
   set" check -- each within-subject fold trains on only ~32 trials, and the
   saved per-epoch training curve will show directly whether the model can
   drive training loss down on that little data.)
2. **Is there any decodable signal in this feature/graph representation for
   a single subject**, decoupled from the (much harder) cross-subject
   generalization problem that LOSO also has to solve?

If within-subject accuracy also sits at chance, the problem is the
features/graph, not cross-subject generalization. If it clears chance
(classic CSP+LDA motor-imagery baselines land around 65-80% within-subject),
the representation does carry signal and the LOSO bottleneck is
specifically about generalizing it across subjects.

This plan is for a **diagnostic tool**, not a replacement for the LOSO
pipeline -- it lives alongside `loso.py`/`hyperparameter_search.py` and
reuses their building blocks rather than duplicating the model/engine layer.

## Constraints that shape the design

- **Only 40 trials per subject** (20/20 class-balanced, confirmed via
  `validate_dataset`'s `subject_counts == 40` check in both combination
  modules). A 5-fold stratified split gives 32 train / 8 eval per fold (4/4
  class balance in the eval fold) -- there isn't enough data left over to
  carve out a *third* split for early-stopping without shrinking training
  data further.
- **No early stopping, fixed epoch budget instead.** Rather than eating into
  the tiny training set for a validation split, each fold trains for a fixed
  `epochs` budget and checkpoints on **best training loss** (zero-leakage --
  it never touches the eval fold, unlike using eval-loss to pick a
  checkpoint). This mirrors the existing `best_validation_loss` checkpoint
  pattern in `train_loso_fold` (`training/loso.py`), just substituting
  training loss since there is no held-out validation split here.
- **Normalization** reuses the existing per-subject `fit_feature_normalization`
  unchanged (mean/std from that subject's *own* trials, including whichever
  fold is held out this run). This is the same scope already documented and
  accepted in `AUDIT.md` item 3 for the main pipeline -- flagged again in the
  new README section so results aren't misread as leak-free by a stricter
  standard than the rest of the package uses. Low risk here specifically
  because it touches no labels and both classes are represented in the
  normalization stats.
- **`repeats` for robustness.** 32-40 trials is a small-sample regime, so a
  single 5-fold split is a noisy estimate. Support an optional `repeats`
  parameter (default 1) that reruns the k-fold split with a different seed
  and aggregates -- cheap to add since it's just re-invoking the same
  fold-construction + training loop.

## Approach

### 1. `data/loso_split.py` -- new split constructor

Add `WithinSubjectFold` (`train_graph_indices`, `evaluation_graph_indices`,
`subject_id`, `fold`) and:

```python
def create_within_subject_folds(
    subject_ids: np.ndarray,
    labels: np.ndarray,
    target_subject: int,
    *,
    k: int = 5,
    seed: int = 42,
) -> tuple[WithinSubjectFold, ...]:
```

Uses `sklearn.model_selection.StratifiedKFold(n_splits=k, shuffle=True,
random_state=seed)` (new import; `GroupKFold` is already used the same way
for `create_inner_cv_folds`) over `target_subject`'s own trial indices only,
stratified by `labels` so every fold keeps the 50/50 class balance. Add
`_validate_within_subject_folds` mirroring `_validate_inner_folds`'s
philosophy: every one of the subject's trials appears in exactly one fold's
evaluation set, train/evaluation disjoint per fold, and no other subject's
indices appear anywhere.

Field name is deliberately `evaluation_graph_indices`, not `validation_...`
-- there is no early-stopping validation split in this design (see
Constraints), so reusing "validation" would misleadingly imply one exists
and risk confusion with the LOSO pipeline's early-stopping validation role.

### 2. Combination modules -- refactor + one new function each

`without_csd_alpha_wpli.py` and `csd_alpha_wpli.py` both already have
`create_group_dataloaders`, which does: resolve subject IDs to graph
indices, fit normalization, build train/validation loaders via the
existing `_make_loader`. Factor the index-to-loaders tail out into a small
shared helper each module already has all the pieces for:

```python
def _create_indexed_dataloaders(
    dataset, train_graph_indices, evaluation_graph_indices, config,
    *, dataset_validated=False,
) -> tuple[DataLoader, DataLoader, FeatureNormalization]:
    if not dataset_validated:
        validate_dataset(dataset)
    band_index = alpha_band_index(dataset)
    normalization = fit_feature_normalization(dataset, epsilon=config.normalization_epsilon)
    train_loader = _make_loader(dataset, train_graph_indices, normalization, config, band_index=band_index, shuffle=True)
    evaluation_loader = _make_loader(dataset, evaluation_graph_indices, normalization, config, band_index=band_index, shuffle=False)
    return train_loader, evaluation_loader, normalization
```

`create_group_dataloaders` becomes a thin wrapper (`_graph_indices_for_subjects`
on both subject-ID tuples, then delegate) -- same public signature and
behavior, so `hyperparameter_search.py` and its tests are unaffected. Add
the new, genuinely different-shaped public function:

```python
def create_within_subject_dataloaders(
    *, dataset, train_graph_indices, evaluation_graph_indices, config,
    dataset_validated=False,
) -> tuple[DataLoader, DataLoader, FeatureNormalization]:
    return _create_indexed_dataloaders(dataset, train_graph_indices, evaluation_graph_indices, config, dataset_validated=dataset_validated)
```

which takes raw graph indices directly (a within-subject fold has no
subject-ID tuple to resolve -- it's already one subject's trial indices).

Register it on `Combination` in `combinations.py` (new
`create_within_subject_dataloaders` field, filled for both entries),
matching the existing registry pattern.

### 3. `training/config.py` -- new `WithinSubjectConfig`

Sibling dataclass to `TrainingConfig`, same `__post_init__` validation
style, but trimmed to what applies (no `patience`, `minimum_improvement`,
`validation_subjects`, or `seed_strategy` -- none of those concepts exist
here):

```python
@dataclass(frozen=True)
class WithinSubjectConfig:
    combination: str = DEFAULT_COMBINATION
    run_name: str | None = None
    overwrite: bool = False
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    gradient_clip_norm: float | None = 1.0
    folds: int = 5
    repeats: int = 1
    num_workers: int = 0
    seed: int = 42
    device: str | None = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    save_outputs: bool = True
```

`epochs=50` matches the existing `search_epochs` default used elsewhere for
smaller-budget runs -- flagged in the plan and README as a starting point to
sanity-check via the training-loss curve on a small pilot, not a validated
number, since the per-fold training set here (~32 trials) is far smaller
than anything else in this package trains on.

### 4. New file: `training/within_subject.py`

Mirrors `hyperparameter_search.py`'s reuse of `engine.py` building blocks
(`train_epoch`, `evaluate`, `set_seed`, `resolve_device`) and `EEGGCN1`:

- `WithinSubjectEpochRecord` (`epoch`, `training: ClassificationMetrics`) --
  lighter than `loso.py`'s `EpochRecord` since there's no validation split
  per epoch here.
- `WithinSubjectFoldResult` (`subject_id`, `fold`, `repeat`, `best_epoch`,
  `epochs_ran`, `evaluation: ClassificationMetrics`,
  `history: tuple[WithinSubjectEpochRecord, ...]`) with an `as_dict()`
  matching the `FoldResult` pattern.
- `train_within_subject_fold(subject_id, fold, *, combination, dataset,
  config, repeat, show_progress) -> WithinSubjectFoldResult`: builds loaders
  via `combination.create_within_subject_dataloaders(...)`, seeds via
  `set_seed(config.seed)`, builds a fresh `EEGGCN1`, trains for the full
  `config.epochs` budget (no early stop), tracking the best-training-loss
  checkpoint (`copy.deepcopy(model.state_dict())`, same pattern as
  `train_loso_fold`) and reloading it before the single `evaluate(...)` call
  on the fold's evaluation loader.
- `train_within_subject(subject_id, config, *, dataset, combination,
  show_progress) -> tuple[WithinSubjectFoldResult, ...]`: builds
  `create_within_subject_folds(..., k=config.folds, seed=config.seed)`,
  optionally repeated `config.repeats` times with `seed + repeat` (each
  repeat reshuffles fold membership), looping `train_within_subject_fold`
  over every (fold, repeat) pair.
- `summarize_within_subject_results(results) -> dict`: two-level
  aggregation mirroring `summarize_results`'s bootstrap-CI machinery
  (`training/loso.py`) but keyed off `.evaluation` instead of `.test`: first
  average each subject's own `(fold, repeat)` results into one per-subject
  balanced accuracy/kappa/etc., then bootstrap-CI *across subjects* on that
  per-subject-mean list -- consistent with `summarize_results`'s existing
  "equal subject weighting" philosophy. Implemented as its own small
  function (not a generalization of `summarize_results`) to avoid touching
  already-tested code for a shape it wasn't designed for.
- `train_all_within_subject(config, subject_ids=None, *, dataset=None,
  show_progress=True) -> tuple[list[WithinSubjectFoldResult], dict]`: loads
  the dataset once (or reuses one passed in, matching `train_all_loso_folds`'s
  pattern), resolves `subject_ids` (defaults to every subject in the
  dataset; accepts an explicit small list for a fast pilot), loops
  `train_within_subject` per subject, aggregates via
  `summarize_within_subject_results`.
- Progress/output: reuse `_bars_enabled`, `_log`, `_write_json`,
  `_git_provenance`, `_package_versions`, `_default_run_name` imported from
  `.loso` rather than duplicating that provenance/logging boilerplate --
  these are generic utilities with no LOSO-specific logic inside them. Saves
  `run_manifest.json`, `subject_XX/within_subject_results.json` (raw
  per-fold/per-repeat results), and a run-level
  `within_subject_summary.json` (per-subject means + the across-subject
  aggregate) when `save_outputs=True`.

### 5. New CLI: `training/within_subject_train.py`

Mirrors `train.py`/`search_train.py`'s `build_parser()` shape:
`--subject N` / `--all-subjects` / `--subjects 1,2,3` (mutually exclusive
group of three, the last for a fast few-subject pilot), `--combination`,
`--folds` (default 5), `--repeats` (default 1), `--epochs` (default 50),
`--batch-size`, `--learning-rate`, `--weight-decay`, `--gradient-clip-norm`
/ `--no-gradient-clipping`, `--seed`, `--device`, `--num-workers`,
`--output-dir`, `--run-name`, `--overwrite`, `--no-save`. No
`--patience`-style flag at all, by design (see Constraints).

### 6. `training/__init__.py`

Export `WithinSubjectConfig` (from `.config`), and
`WithinSubjectFoldResult`, `train_within_subject`, `train_all_within_subject`,
`summarize_within_subject_results` (from `.within_subject`), added to
imports and `__all__` alongside the existing exports.

### 7. Tests

- `data/test_loso_split.py`: new `CreateWithinSubjectFoldsTests` --
  partition covers the target subject's every trial exactly once across
  folds' evaluation sets, train/evaluation disjoint per fold, each
  evaluation fold stays class-balanced (stratification actually worked),
  reproducible given a fixed seed, different seeds reshuffle fold
  membership, rejects `k` larger than the subject's trial count, rejects a
  `target_subject` absent from `subject_ids`, and confirms no other
  subject's indices ever appear.
- New `training/test_within_subject.py`: integration test (matching
  `test_hyperparameter_search.py`'s style) with a tiny budget (`folds=2`,
  `epochs=2`, `save_outputs=False`) on one real subject -- asserts it
  completes, returns the expected number of `WithinSubjectFoldResult`s,
  every metric is finite and in `[0, 1]`, same-seed reproducibility, and
  that `repeats=2` produces `2 * folds` results with different fold
  membership across repeats.
- Full suite + a real small pilot CLI run (`--subjects 1,2 --folds 3
  --epochs 10 --no-save`) to eyeball sane output and inspect the saved
  training-loss curve for signs of the model actually fitting (or not) its
  ~30-trial training folds.

### 8. `training/README.md` and `AUDIT.md`

Add a "Within-subject classification (diagnostic)" section to the README:
purpose (decouple cross-subject generalization from "is there signal at
all"), usage examples, flag table, output files, the normalization-scope
note repeated from Constraints, and interpretation guidance (chance is
exactly 0.5 balanced accuracy by construction since folds are
class-stratified; classic CSP+LDA within-subject baselines land ~65-80% for
motor imagery, useful as a rough external reference point). `AUDIT.md` gets
one short pointer sentence in the Bottom line section noting this
diagnostic tool now exists (it is not a protocol-compliance item, so no
table rows change).

## What does NOT change

- `loso.py`, `hyperparameter_search.py`, `search_train.py`, `train.py`, and
  every existing test are untouched except the internal-only
  `create_group_dataloaders` refactor in the two combination modules, whose
  public signature and behavior stay identical.
- No model/architecture changes -- same `EEGGCN1`/`EEGGCN1Config`, so a
  within-subject signal finding (or lack of one) is attributable to the
  data/task, not a different model.

## Verification (once implemented)

1. `python -m pytest src/ourexperimentversionfour -q` -- all existing +
   new tests pass.
2. A real pilot run: `python -m src.ourexperimentversionfour.training.within_subject_train
   --subjects 1,2 --folds 3 --epochs 10 --no-save` -- inspect the printed
   per-fold evaluation metrics and training-loss curve for sanity (finite
   values, training loss actually decreasing, plausible balanced-accuracy
   range).
3. Do **not** run the full 50-subject sweep as part of verification --
   confirm the small pilot works first, then it's your call whether/when to
   run all 50 (much cheaper than the hyperparameter search: 50 subjects x 5
   folds x 50 epochs, no inner search multiplier, so this should finish in
   well under an hour based on this session's per-epoch timings).
