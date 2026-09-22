# Training: LOSO GCN on selectable Graph-Liu2024-VersionTwo combinations

Leave-one-subject-out (LOSO) training/evaluation for `EEGGCN1` (a sparse,
edge-weighted GCN) on one of the registered feature combinations in
`src/ourexperimentversionfive/data/combinations.py`. Which combination is
trained is a config/CLI choice, not a code change -- see `--combination`
below.

All commands assume you're running from the repository root
(`/workspaces/EEG`) so `python -m ...` can resolve the package.

## Prerequisites

- The dataset must already exist at `data/moabb/Graph-Liu2024-VersionTwo`
  (built via `src/datautils/graphdataversiontwo`; see that package's README
  if it's missing).
- Dependencies from `requirements.txt` installed (`torch`,
  `torch-geometric`, `numpy`, `tqdm`, `scikit-learn`, ...).

## Quick reference

```bash
# One held-out subject, printed to stdout, nothing saved to disk.
python -m src.ourexperimentversionfive.training.train \
  --test-subject 1 --epochs 10 --no-save

# Full LOSO sweep (all 50 subjects), saved under a named run directory.
# --no-early-stopping runs the full --epochs budget on every fold instead of
# letting early stopping cut folds short.
python -m src.ourexperimentversionfive.training.train \
  --all-subjects --epochs 200 --no-early-stopping --run-name my_run

# Same, but on the CSD combination instead of the default without-CSD one.
python -m src.ourexperimentversionfive.training.train \
  --combination csd_alpha_wpli --all-subjects --epochs 200 --run-name csd_run

# Same held-out-subject check, but with SGD+momentum instead of the default
# AdamW -- useful for comparing optimizer behavior on the same data/model.
python -m src.ourexperimentversionfive.training.train \
  --test-subject 1 --epochs 10 --optimizer sgd --momentum 0.9 --no-save
```

## Choosing a combination

`--combination {without_csd_alpha_wpli, csd_alpha_wpli}` (default:
`without_csd_alpha_wpli`). Both share the same 29-node / 8-feature /
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
| `--node-normalization {none,zscore}` | `none` | Optional training-only node scaling. |
| `--output-dir PATH` | `src/ourexperimentversionfive/outputs` | |
| `--run-name NAME` | UTC timestamp | Output subdirectory name. Reusing a non-empty name fails unless `--overwrite` is also passed. |
| `--overwrite` | off | Allow writing into an existing non-empty run directory. |
| `--no-save` | off | Skip all output artifacts; only the JSON result/summary is printed to stdout. Useful for quick checks. |

## What gets saved

With `--no-save` omitted, a run directory (`<output-dir>/<run-name>/`)
contains:

- `run_manifest.json` -- config, resolved device, combination, dataset
  provenance, git commit; `status` flips `running` -> `completed`.
- `subject_XX/best_model.pt` -- best-validation-loss checkpoint: model
  weights, configs, optional training-only feature normalization, split, and result.
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

## Node feature selection and optional normalization

Schema-2 source arrays contain eight columns. This experiment selects exactly
`delta_power_db`, `theta_power_db`, `alpha_power_db`, `beta_power_db`,
`gamma_power_db`, `spectral_entropy`, `hjorth_mobility`, and `hjorth_complexity`
(by validated metadata names; source indices 0–7). Entropy stays over
1–40 Hz and Hjorth uses the broadband trial signal. EEG and CSD use their own
node/edge variants. Model inputs are `(29, 8)` and alpha-wPLI attributes are
`(812, 1)`; reverse edge entries retain identical weights for message passing.

All entry points accept `--node-normalization {none,zscore}`:

- `none` (default): use selected saved values directly. No statistics are fitted.
- `zscore`: fit eight means/stds from each fold's training trials and electrodes
  only, with standard deviations floored at `1e-8`. Apply the same statistics to
  validation/test trials, including previously unseen subjects. All eight node
  columns are scaled; edge weights are unchanged.

Inner search refits inside each training fold; final training refits on its own
training partition. Within-subject fitting excludes evaluation trials. The mode
stays fixed across search and final refitting and is recorded in configuration,
manifests, and fold results/checkpoints. No held-out-subject statistics are used.
`fit_feature_normalization(..., mode="zscore")` requires explicit training indices.
Disabled normalization uses an explicit state with empty statistics.

Example:

```bash
python -m src.ourexperimentversionfive.training.train --test-subject 1 \
  --node-normalization none --run-name alpha_raw
python -m src.ourexperimentversionfive.training.train --test-subject 1 \
  --node-normalization zscore --run-name alpha_zscore
```

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
python -m src.ourexperimentversionfive.training.search_train \
  --test-subject 1 --inner-folds 2 --search-epochs 5 --epochs 10 --no-save

# Full nested-CV sweep (default 10-fold x 12-candidate search per subject).
python -m src.ourexperimentversionfive.training.search_train \
  --all-subjects --epochs 200 --no-early-stopping --run-name my_search_run

# SGD+momentum instead of AdamW -- applies to both the inner-CV search and
# the final retrain (unlike --learning-rate/--weight-decay, see note above).
python -m src.ourexperimentversionfive.training.search_train \
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
python -m src.ourexperimentversionfive.training.within_subject_cli \
  --subject 1 --folds 3 --epochs 10 --no-save

# A few-subject pilot, saved to a named run directory.
python -m src.ourexperimentversionfive.training.within_subject_cli \
  --subjects 1,2,3 --run-name within_subject_pilot

# Every subject in the dataset.
python -m src.ourexperimentversionfive.training.within_subject_cli \
  --all-subjects --run-name within_subject_full

# Same small pilot, but with SGD+momentum instead of the default AdamW --
# useful for comparing optimizer behavior on the same diagnostic.
python -m src.ourexperimentversionfive.training.within_subject_cli \
  --subject 1 --folds 3 --epochs 10 --optimizer sgd --momentum 0.9 --no-save
```

Each subject's own 40 trials are split into `--folds` class-stratified
folds (default 5: 32 train / 8 evaluation per fold, 4/4 class balance).
Each fold trains **twice**: a selection pass early-stops on an inner
validation split carved out of that fold's own training trials (never its
evaluation trials -- there isn't enough of a subject's 40 trials to carve
out a third split without this being an issue, so the inner split is
reused-and-discarded per fold rather than held out permanently) to pick an
epoch count, then a final pass fits *all* of that fold's training trials
for exactly that many epochs before the one, real evaluation. This
replaced an earlier design that checkpointed on **training** loss instead
of a validation signal, which -- since training loss falls
near-monotonically -- always selected the most-overfit, latest epoch and
produced chance-level held-out accuracy despite >90% training accuracy.
`--repeats N` reruns the fold split with a different seed and aggregates,
since a single 5-fold split over ~40 trials is a noisy estimate.

| Flag | Default | Notes |
|---|---|---|
| `--subject N` / `--all-subjects` / `--subjects 1,2,3` | — | Mutually exclusive group of three; one is required. `--subjects` is a comma-separated list for a fast few-subject pilot. |
| `--combination NAME` | `without_csd_alpha_wpli` | Which registered combination to train on. |
| `--folds N` | `5` | Within-subject stratified k-fold count. |
| `--repeats N` | `1` | Re-run the k-fold split this many times with a different seed and aggregate. |
| `--epochs N` | `50` | Selection-pass epoch cap; the final pass runs however many epochs the selection pass chose. Matches the existing `search_epochs` default used elsewhere for smaller-budget runs; a starting point to sanity-check via the training-loss curve, not a validated number. |
| `--batch-size N` | `8` | Deliberately well below every fold's smallest training-trial count (24-32) -- at `TrainingConfig`'s 32 default, every fold's entire training set fit in one batch, so every "epoch" was a single full-batch gradient step, which combined with Adam's unstable early moment estimates caused a large fraction of folds to early-stop after just one step on pure noise (verified: >40% of folds finished final training at or below 60% *training* accuracy at batch size 32). 8 gives 3-4 mini-batches per epoch for every fold. |
| `--learning-rate LR` / `--weight-decay WD` | `0.001` / `0.0005` | Learning rate lowered from the original `0.01` Kipf & Welling default; see the smaller-classifier comparison below. |
| `--optimizer {adamw,adam,sgd}` | `adamw` | Same shared choice as `train.py`/`search_train.py` (see `engine.build_optimizer`). |
| `--momentum X` | `0.9` | SGD momentum; ignored for `adamw`/`adam`. |
| `--patience N` / `--no-early-stopping` | `None` (no early stopping) | Early-stopping patience on the inner-validation-split loss used for epoch selection; disabled by default so every fold runs the full `--epochs` budget (see `OVERFITTING_AUDIT.md` on 8-trial-validation selection noise). |
| `--classifier {mlp,linear}` | `linear` | See "Smaller classifier comparison" below. |
| `--edge-mode {weighted,self_only}` | `self_only` | See "Self-only adjacency comparison" below. |
| `--node-normalization {none,zscore}` | `zscore` | Train-only feature standardization; see "Normalization" below. |
| `--minimum-improvement X` | `0.0001` | |
| `--inner-validation-fraction X` | `0.25` | Fraction of each fold's training trials held back for epoch selection. |
| `--gradient-clip-norm X` / `--no-gradient-clipping` | `1.0` | Mutually exclusive pair. |
| `--seed N` | `42` | Base seed; each repeat uses `seed + repeat`. |
| `--device {cpu,cuda}` | auto-detect | |
| `--num-workers N` | `0` | |
| `--output-dir PATH` / `--run-name NAME` / `--overwrite` | see `train.py` | Same semantics as `train.py`. |
| `--no-save` | off | Skip all output artifacts; only JSON is printed to stdout. |

**Normalization:** the same optional mode applies here. `none` performs no
fitting. `zscore` fits only from the current fold's training indices, as in LOSO
and inner search. Per-fold results record the mode, training/evaluation indices,
and fitted means/stds when enabled.

**Output** (with `--no-save` omitted): `run_manifest.json` (config, resolved
combination, dataset provenance, git commit), `subject_XX/within_subject_results.json`
(raw per-fold/per-repeat results and training-loss history), and a run-level
`within_subject_summary.json` (per-subject means plus a bootstrap-CI
aggregate across subjects, mirroring `summarize_results`'s equal-subject-
weighting philosophy).

## Verification scope

Version five is a new eight-feature experiment. Copied version-four scores do not
establish its performance. See `../VERSION_FIVE_IMPLEMENTATION_PLAN.md` for the
migration and verification record. The historical documents are explicitly labeled;
full cross-validation and hyperparameter sweeps must be run separately before
reporting version-five performance.

## Within-subject overfitting diagnostics

New runs save `final_training` in each fold result: training-set metrics from
one evaluation-mode pass over the final model, after held-out evaluation.
Dropout is disabled and BatchNorm running statistics stay frozen, so these
metrics can be compared directly with that fold's `evaluation` metrics.
The existing final-pass `history[].training` remains online optimization
metrics, with dropout enabled and weights changing during the epoch.

`generalization_gap.accuracy` and `.balanced_accuracy` mean training minus
held-out score; `.loss` means held-out minus training loss. Positive values
indicate worse held-out performance. A gap is a diagnostic, not an automatic
proof of overfitting or a new selection criterion.

`inner_selection.history` now saves every attempted selection epoch's
`training_online` and `validation` metrics, including epochs after the best
one. Inner training/validation graph indices are also saved for auditing.
These training curves remain online metrics, explicitly labeled as such.
The run summary's `diagnostics` contains mean final training metrics and
mean gaps, first averaging folds/repeats per subject, then subjects equally.
Old saved runs are not backfilled.

Use a new run name and keep your normalized comparison's settings, for example:

```bash
python -m src.ourexperimentversionfive.training.within_subject_cli \
  --all-subjects --node-normalization zscore --batch-size 8 \
  --epochs 50 --no-early-stopping --seed 42 \
  --run-name within_subject_zscore_diagnostics
```

### Self-only adjacency comparison

Within-subject CLI accepts `--edge-mode {weighted,self_only}` (default:
`self_only`). `self_only` replaces connectivity inside the model with one
unit self-loop per electrode in both selection and final training. The GCN
therefore applies its learned feature transform independently at each node;
BatchNorm and the flattened classifier still combine information across nodes.
Parameter count, dataset arrays, splits, and feature scaling are unchanged.
The mode is recorded in the run manifest/configuration.

```bash
python -m src.ourexperimentversionfive.training.within_subject_cli \
  --all-subjects --node-normalization zscore --edge-mode self_only \
  --batch-size 8 --epochs 50 --no-early-stopping --seed 42 \
  --run-name within_subject_full_zscore_self_only
```

Compare evaluation metrics and `diagnostics` against
`within_subject_full_zscore_evaluation_mode`. Epoch selection can choose a
different budget because the representation changed; its rule stays identical.

### Smaller classifier comparison

`--classifier linear` (now the default) selects a direct `464 → 2` head,
removing the hidden classifier layer and its activation/dropout. The GCN,
BatchNorm, and post-GCN activation/dropout remain. Total trainable
parameters fall from 13,721 to 1,106 versus `--classifier mlp`'s
`464 → 29 → 2` head. Both selection and final training use the chosen head;
the run manifest/configuration records it. This option is exposed by the
within-subject CLI.

```bash
python -m src.ourexperimentversionfive.training.within_subject_cli \
  --all-subjects --node-normalization zscore --edge-mode self_only \
  --classifier linear --learning-rate 0.001 --batch-size 8 \
  --epochs 50 --no-early-stopping --seed 42 \
  --run-name within_subject_full_zscore_self_only_lr0001_linear
```

Compare against `within_subject_full_zscore_self_only_lr0001`, keeping the
same selection rule and training settings. This comparison changes both
classifier capacity and the hidden head's nonlinearity/dropout; it does not
isolate parameter count alone.
