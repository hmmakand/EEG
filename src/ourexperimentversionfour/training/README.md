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
```

Same base flags as `train.py` (`--combination`, `--epochs`, `--patience`,
etc. -- these apply to the *final retrain*), plus:

| Flag | Default | Notes |
|---|---|---|
| `--inner-folds N` | `10` | Grouped inner-CV fold count over the 49 development subjects; fixed at 10 to match the specified 44-45/4-5 split shape. |
| `--search-epochs N` | `50` | Max epochs per inner-CV training run (separate from `--epochs`, the final retrain's budget). |
| `--search-patience N` / `--no-search-early-stopping` | `8` | Early-stopping patience for inner-CV runs specifically; independent of `--patience`. |

Saved output adds `subject_XX/hyperparameter_search.json` (selected
hyperparameters + every candidate's mean/per-fold balanced accuracy) per
fold, and `selected_hyperparameters_by_subject.json` at the run level for
`--all-subjects`.

## Known baseline result

As of this standardized baseline (standard GCN hyperparameters above, 200
epochs, per-subject normalization), LOSO training does **not** clear chance
level (~50% accuracy, AUC ~0.5) on either combination when pooling all
training subjects -- confirmed to not be a data or gradient-flow bug (see
the diagnostics in this package's history), and consistent with the
model needing richer features or a subject-adaptive architecture, not
just more epochs or better-tuned hyperparameters. Treat this as the
starting point for future improvement, not a working classifier yet.
