# Training: LOSO GCN on selectable Graph-Liu2024-VersionOne combinations

Leave-one-subject-out (LOSO) training/evaluation for `EEGGCN1` (a sparse,
edge-weighted GCN) on one of the registered feature combinations in
`src/ourexperimentversionfour/data/combinations.py`. Which combination is
trained is a config/CLI choice, not a code change -- see `--combination`
below.

All commands assume you're running from the repository root
(`/workspaces/EEG`) so `python -m ...` can resolve the package.

## Prerequisites

- The dataset must already exist at `data/moabb/Graph-Liu2024-VersionOne`
  (built via `src/datautils/graphdataversionone`; see that package's README
  if it's missing).
- Dependencies from `requirements.txt` installed (`torch`,
  `torch-geometric`, `numpy`, `tqdm`, `scikit-learn`, ...).

## Quick reference

```bash
# One held-out subject, printed to stdout, nothing saved to disk.
python -m src.ourexperimentversionfour.training.train \
  --test-subject 1 --epochs 10 --no-save

# Full LOSO sweep (all 50 subjects), saved under a named run directory.
# --no-early-stopping runs the full --epochs budget on every fold instead of
# letting early stopping cut folds short.
python -m src.ourexperimentversionfour.training.train \
  --all-subjects --epochs 200 --no-early-stopping --run-name my_run

# Same, but on the CSD combination instead of the default without-CSD one.
python -m src.ourexperimentversionfour.training.train \
  --combination csd_alpha_wpli --all-subjects --epochs 200 --run-name csd_run

# Same held-out-subject check, but with SGD+momentum instead of the default
# AdamW -- useful for comparing optimizer behavior on the same data/model.
python -m src.ourexperimentversionfour.training.train \
  --test-subject 1 --epochs 10 --optimizer sgd --momentum 0.9 --no-save
```

## Choosing a combination

`--combination {without_csd_alpha_wpli, csd_alpha_wpli}` (default:
`without_csd_alpha_wpli`). Both share the same 29-node / 6-feature /
single-edge-weight contract, so nothing else about the command changes when
you switch. See `combinations.py` for the registry; adding a new
combination there makes it available here automatically.

## Flag reference

| Flag | Default | Notes |
|---|---|---|
| `--test-subject N` | — | Train/evaluate one held-out subject. Mutually exclusive with `--all-subjects`; one of the two is required. |
| `--all-subjects` | — | Run all 50 LOSO folds (one per subject) and print the aggregate summary. |
| `--combination NAME` | `without_csd_alpha_wpli` | Which registered combination to train on. |
| `--epochs N` | `200` | Max epochs per fold; early stopping can end a fold sooner. |
| `--batch-size N` | `32` | |
| `--learning-rate LR` | `0.01` | Standard Adam/GCN default (Kipf & Welling 2017), not tuned to any specific combination. |
| `--weight-decay WD` | `0.0005` | |
| `--optimizer {adamw,adam,sgd}` | `adamw` | Which optimizer to train with -- a training-mechanic choice, shared by `train.py`/`search_train.py`/`within_subject_cli.py` alike (see `engine.build_optimizer`), independent of the data split. |
| `--momentum X` | `0.9` | SGD momentum; ignored for `adamw`/`adam`. |
| `--patience N` / `--no-early-stopping` | `10` | Mutually exclusive pair. Patience is the number of epochs without validation-loss improvement before stopping a fold early; `--no-early-stopping` disables it entirely, so every fold always runs the full `--epochs` budget. |
| `--minimum-improvement X` | `0.0001` | Minimum validation-loss delta counted as improvement. |
| `--gradient-clip-norm X` / `--no-gradient-clipping` | `1.0` | Mutually exclusive pair; the second disables clipping entirely. |
| `--validation-subjects N` | `5` | Subjects held out of training (per fold) for early stopping / model selection. |
| `--seed N` | `42` | Base seed. |
| `--seed-strategy {shared,per_fold}` | `shared` | `shared` reuses one init seed for every fold; `per_fold` derives `seed + test_subject_id`. |
| `--device {cpu,cuda}` | auto-detect | |
| `--num-workers N` | `0` | DataLoader worker processes. |
| `--output-dir PATH` | `src/ourexperimentversionfour/outputs` | |
| `--run-name NAME` | UTC timestamp | Output subdirectory name. Reusing a non-empty name fails unless `--overwrite` is also passed. |
| `--overwrite` | off | Allow writing into an existing non-empty run directory. |
| `--no-save` | off | Skip all output artifacts; only the JSON result/summary is printed to stdout. Useful for quick checks. |

## What gets saved

With `--no-save` omitted, a run directory (`<output-dir>/<run-name>/`)
contains:

- `run_manifest.json` -- config, resolved device, combination, dataset
  provenance, git commit; `status` flips `running` -> `completed`.
- `subject_XX/best_model.pt` -- best-validation-loss checkpoint: model
  weights, configs, per-subject feature normalization, split, and result.
- `subject_XX/result.json` -- the same result/split/normalization info as
  plain JSON (no tensors).
- `loso_results.json` / `loso_results.csv` -- only for `--all-subjects`:
  per-fold results plus the mean/std summary across folds.

Every `ClassificationMetrics` record (per-epoch history, `test`, and CSV
rows) includes `balanced_accuracy` (mean per-class recall) and
`cohens_kappa` (chance-corrected agreement) alongside the existing
loss/accuracy/f1/recall/precision/auc. The `loso_results.json` summary
additionally reports a 95% percentile-bootstrap confidence interval for
every metric across the 50 folds (`ci95_low_*`/`ci95_high_*`, alongside
`mean_*`/`std_*`) -- see `summarize_results` in `loso.py`.

## Normalization baseline

Node features are z-scored **per subject**, using only that subject's own
raw (unlabeled) trials -- not one global mean/std pooled across training
subjects. This removes each subject's absolute baseline (skull thickness,
electrode contact, alertness, ...) the same way for train, validation, and
the held-out test subject, and is leakage-safe since no labels are involved
(standard cross-subject EEG/BCI alignment practice). See
`without_csd_alpha_wpli.fit_feature_normalization` / `FeatureNormalization`.

## Nested-CV hyperparameter search

`search_train.py` is a separate entry point (not a flag on `train.py`)
because its cost profile is qualitatively different: for each outer fold, it
first runs grouped 10-fold inner cross-validation over the 49 development
subjects to select `learning_rate`/`weight_decay` (ranked by mean validation
balanced accuracy -- see `AUDIT.md` item 2 for why), *then* retrains on the
44/5 split and evaluates the held-out subject exactly once, same as
`train.py`. Default budget is 10 inner folds x 12 candidates = 120 extra
training runs per outer fold before the final retrain -- realistically an
overnight-or-longer run for `--all-subjects` at the default budget; always
validate on 1-2 subjects first.

```bash
# One subject, small search budget, nothing saved -- for trying it out.
python -m src.ourexperimentversionfour.training.search_train \
  --test-subject 1 --inner-folds 2 --search-epochs 5 --epochs 10 --no-save

# Full nested-CV sweep (default 10-fold x 12-candidate search per subject).
python -m src.ourexperimentversionfour.training.search_train \
  --all-subjects --epochs 200 --no-early-stopping --run-name my_search_run

# SGD+momentum instead of AdamW -- applies to both the inner-CV search and
# the final retrain (unlike --learning-rate/--weight-decay, see note above).
python -m src.ourexperimentversionfour.training.search_train \
  --test-subject 1 --inner-folds 2 --search-epochs 5 --epochs 10 \
  --optimizer sgd --momentum 0.9 --no-save
```

Same base flags as `train.py` (`--combination`, `--epochs`, `--patience`,
etc. -- these apply to the *final retrain*), plus:

Note: `--learning-rate`/`--weight-decay` set the *base* config, but the
search always overrides exactly those two fields for the final retrain (see
the CLI's own `--help` epilog) -- passing them has no effect on the actual
result. `--optimizer`/`--momentum` are different: the grid never searches
over them, so whatever you pass applies to *both* the inner-CV search and
the final retrain.

| Flag | Default | Notes |
|---|---|---|
| `--inner-folds N` | `10` | Grouped inner-CV fold count over the 49 development subjects; fixed at 10 to match the specified 44-45/4-5 split shape. |
| `--search-epochs N` | `50` | Max epochs per inner-CV training run (separate from `--epochs`, the final retrain's budget). |
| `--search-patience N` / `--no-search-early-stopping` | `8` | Early-stopping patience for inner-CV runs specifically; independent of `--patience`. |

Saved output adds `subject_XX/hyperparameter_search.json` (selected
hyperparameters + every candidate's mean/per-fold balanced accuracy) per
fold, and `selected_hyperparameters_by_subject.json` at the run level for
`--all-subjects`.

## Within-subject classification (diagnostic)

`within_subject_cli.py` is a separate diagnostic tool, not a mode of
`train.py`/`search_train.py` -- see `WITHIN_SUBJECT_PLAN.md` for the full
rationale. It trains and evaluates on **one subject's own trials only**,
with no cross-subject generalization involved, to isolate two questions the
LOSO number conflates: can the model fit this data at all (an implicit
"overfit a small set" check), and is there any decodable signal in this
feature/graph representation for a single subject, decoupled from the
harder cross-subject problem LOSO also has to solve.

```bash
# One subject, small budget, nothing saved -- for trying it out.
python -m src.ourexperimentversionfour.training.within_subject_cli \
  --subject 1 --folds 3 --epochs 10 --no-save

# A few-subject pilot, saved to a named run directory.
python -m src.ourexperimentversionfour.training.within_subject_cli \
  --subjects 1,2,3 --run-name within_subject_pilot

# Every subject in the dataset.
python -m src.ourexperimentversionfour.training.within_subject_cli \
  --all-subjects --run-name within_subject_full

# Same small pilot, but with SGD+momentum instead of the default AdamW --
# useful for comparing optimizer behavior on the same diagnostic.
python -m src.ourexperimentversionfour.training.within_subject_cli \
  --subject 1 --folds 3 --epochs 10 --optimizer sgd --momentum 0.9 --no-save
```

Each subject's own 40 trials are split into `--folds` class-stratified
folds (default 5: 32 train / 8 evaluation per fold, 4/4 class balance).
There is **no early stopping and no validation split** -- each fold trains
for the full `--epochs` budget and checkpoints on best **training** loss
instead of validation loss (zero-leakage, since it never touches the
evaluation fold; there isn't enough of a subject's 40 trials to carve out a
third split without shrinking training data further). `--repeats N` reruns
the fold split with a different seed and aggregates, since a single 5-fold
split over ~40 trials is a noisy estimate.

| Flag | Default | Notes |
|---|---|---|
| `--subject N` / `--all-subjects` / `--subjects 1,2,3` | — | Mutually exclusive group of three; one is required. `--subjects` is a comma-separated list for a fast few-subject pilot. |
| `--combination NAME` | `without_csd_alpha_wpli` | Which registered combination to train on. |
| `--folds N` | `5` | Within-subject stratified k-fold count. |
| `--repeats N` | `1` | Re-run the k-fold split this many times with a different seed and aggregate. |
| `--epochs N` | `50` | Fixed per-fold budget -- no early stopping, so every fold always runs the full budget. Matches the existing `search_epochs` default used elsewhere for smaller-budget runs; a starting point to sanity-check via the training-loss curve, not a validated number. |
| `--batch-size N` | `32` | |
| `--learning-rate LR` / `--weight-decay WD` | `0.01` / `0.0005` | |
| `--optimizer {adamw,adam,sgd}` | `adamw` | Same shared choice as `train.py`/`search_train.py` (see `engine.build_optimizer`). |
| `--momentum X` | `0.9` | SGD momentum; ignored for `adamw`/`adam`. |
| `--gradient-clip-norm X` / `--no-gradient-clipping` | `1.0` | Mutually exclusive pair. |
| `--seed N` | `42` | Base seed; each repeat uses `seed + repeat`. |
| `--device {cpu,cuda}` | auto-detect | |
| `--num-workers N` | `0` | |
| `--output-dir PATH` / `--run-name NAME` / `--overwrite` | see `train.py` | Same semantics as `train.py`. |
| `--no-save` | off | Skip all output artifacts; only JSON is printed to stdout. |

**Normalization** is fit strictly from each fold's **training** indices --
`fit_feature_normalization(..., graph_indices=fold.train_graph_indices)` --
never from that fold's held-out evaluation trials. This is *stricter* than
the main LOSO pipeline's normalization, which fits from a subject's whole
trial set including the held-out portion (documented and accepted in
`AUDIT.md` item 3); the within-subject diagnostic doesn't carry that
deviation, since it is also meant as a verification signal for the
dataset-generation pipeline and the extra strictness costs nothing here (one
subject's ~32 training trials is cheap to refit per fold).

**Output** (with `--no-save` omitted): `run_manifest.json` (config, resolved
combination, dataset provenance, git commit), `subject_XX/within_subject_results.json`
(raw per-fold/per-repeat results and training-loss history), and a run-level
`within_subject_summary.json` (per-subject means plus a bootstrap-CI
aggregate across subjects, mirroring `summarize_results`'s equal-subject-
weighting philosophy).

**Interpretation:** chance is exactly 0.5 balanced accuracy by construction
(folds are class-stratified). Classic CSP+LDA within-subject baselines for
motor imagery land around 65-80% -- a useful rough external reference
point, not a claim that this architecture should match it. If within-subject
accuracy also sits at chance, the bottleneck is the features/graph
representation itself, not cross-subject generalization; if it clears
chance, the representation does carry signal and the LOSO bottleneck is
specifically about generalizing it across subjects.

## Known baseline result

As of this standardized baseline (standard GCN hyperparameters above, 200
epochs, per-subject normalization), LOSO training does **not** clear chance
level (~50% accuracy, AUC ~0.5) on either combination when pooling all
training subjects -- confirmed to not be a data or gradient-flow bug (see
the diagnostics in this package's history), and consistent with the
model needing richer features or a subject-adaptive architecture, not
just more epochs or better-tuned hyperparameters. Treat this as the
starting point for future improvement, not a working classifier yet.
