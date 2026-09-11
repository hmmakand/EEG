# Is the within-subject split a standard, pure within-subject protocol?

Status: **verified against the field-standard reference implementation, and
the one deviation found (normalization scope) has since been fixed** -- see
the "RESOLVED" note below. This
file previously held the implementation plan for the within-subject
diagnostic (now built -- see `training/within_subject.py`,
`training/within_subject_cli.py`, `data/within_subject_split.py`). It is
repurposed here (temporary/scratch file, safe to overwrite) to answer a
follow-up question raised during review: since this diagnostic is meant to
double as a **verification signal for the dataset-generation pipeline**, is
its split actually a standard, uncompromised within-subject protocol, or
does having "folds" mean it's secretly doing something else?

## Short answer

**Yes, it is a standard within-subject protocol**, and not an invented one --
it matches MOABB's own canonical `WithinSessionEvaluation` almost exactly
(MOABB is the toolkit this project's Liu2024 dataset is built from). There is
exactly one documented, low-risk deviation from a maximally strict protocol
(normalization scope), called out below rather than left implicit.

## "Within-subject" and "k-fold" answer two different questions

These get conflated, which is presumably where the "why does within-subject
have folds?" question comes from:

- **Within-subject** is a statement about *data scope*: no other subject's
  trials ever enter this subject's training set or evaluation set. This is
  the property that makes it "pure," and it is fully intact here -- enforced
  by construction and unit-tested (`data/test_within_subject_split.py`):
  every trial in every fold belongs to the one target subject, train and
  evaluation indices are disjoint per fold, and every one of that subject's
  trials appears in exactly one fold's evaluation set.
- **k-fold** is a statement about *evaluation protocol*: how you turn one
  subject's limited trials into a performance estimate. It is not in tension
  with "within-subject" -- it is simply cross-validation run *inside* the
  boundary that within-subject already establishes.

## Why k-fold specifically, not a single train/test split

Checked directly against this project's own data:

```
sample = {'subject': 1, 'trial_index': 0, ..., 'source_file':
  '.../sub-01/eeg/sub-01_task-motor-imagery_eeg.edf', ...}
```

There is no `session` field, and every trial for a subject maps to the same
single `source_file` -- Liu2024, as loaded here, is **one session per
subject, 40 trials total** (20/20 class-balanced). That rules out the
"cleanest" alternative (train on one session, test on another, as e.g. BCI
Competition IV-2a does across its two recorded sessions) -- there is only one
session to split.

With only 40 trials and no second session, a single 80/20 split would (a)
throw away most of the subject's data for either training or evaluation and
(b) yield exactly one noisy point estimate per subject. Stratified k-fold CV
is the field-standard way around this: every trial serves as both training
data (in the folds where it isn't held out) and evaluation data (in the one
fold where it is), the class balance is preserved in every fold via
stratification, and the estimate is far less dependent on the luck of one
particular split.

## Confirmed against the actual reference implementation, not just literature

Rather than relying on a general impression of "this is common practice," I
checked the MOABB package actually installed in this environment
(`moabb==1.5.0`, `pip show moabb`), since it's the toolkit this project's own
Liu2024 dataset is derived from
(`src/datautils/graphdataversionone`). MOABB's own canonical within-subject
evaluation is `moabb.evaluations.WithinSessionEvaluation`, whose splitter is:

```python
# moabb/evaluations/evaluations.py, WithinSessionEvaluation._create_splitter
WithinSessionSplitter(
    n_folds=5,
    shuffle=True,
    random_state=self.random_state,
    cv_class=StratifiedKFold,
)
```

`WithinSessionSplitter.split()` groups by `(subject, session)` and, within
each group, runs `StratifiedKFold(n_splits=5, shuffle=True,
random_state=...)` -- i.e. exactly stratified, shuffled 5-fold CV over one
subject's own trials in one session. Since Liu2024 here has exactly one
session per subject, MOABB's own standard protocol for this dataset
collapses to precisely what `create_within_subject_folds` already does:

| | MOABB `WithinSessionSplitter` (default) | This project's `create_within_subject_folds` |
|---|---|---|
| Fold count | `n_folds=5` | `k=5` (default) |
| Splitter | `StratifiedKFold` | `StratifiedKFold` |
| Shuffle | `True` | `True` |
| Scope | one subject, one session | one subject (single-session dataset) |
| Cross-subject leakage | none (grouped by subject first) | none (validated, unit-tested) |

This is not a coincidental resemblance to defend -- it is the same
mechanism as the field's own reference tool. `repeats` (re-running the
5-fold split with a different seed and aggregating) is an addition beyond
MOABB's default single pass, included because 40 trials is a genuinely small
sample and a single 5-fold draw is a noisy estimate; repeated k-fold for
small-N evaluation is itself a standard variance-reduction technique, not a
departure from the protocol MOABB runs by default.

## The one deviation, stated plainly (not buried) -- RESOLVED

**Update:** implemented. `fit_feature_normalization` (both combination
modules) now takes an optional `graph_indices` parameter, and
`train_within_subject_fold` (`training/within_subject.py`) fits it from
`fold.train_graph_indices` only for every fold. Every other caller
(`create_loso_dataloaders`, `create_group_dataloaders`, LOSO,
hyperparameter search) is unaffected -- omitting `graph_indices` keeps their
exact prior behavior. Verified: full suite (94 tests, 6 new covering the
scoping directly) passes; a real CLI run's numbers changed relative to the
pre-fix run (proving the stricter path is active, not a silent no-op); and
timing stayed ~10s for one subject (no regression -- see
`training/README.md`'s within-subject Normalization note for the updated
description). The paragraph below is kept as the original problem
statement for context.

---

Per-subject feature normalization (`fit_feature_normalization`) was
computed from **all 40 of the subject's trials** -- the training fold plus
the held-out evaluation fold together -- not from the training fold alone. A
maximally strict protocol would refit normalization per fold, from that
fold's ~32 training trials only.

This is not a new or hidden choice: it is the exact same scope already
documented for the LOSO pipeline (`AUDIT.md` item 3, `training/README.md`'s
"Normalization baseline" section) -- z-scoring uses a subject's own
unlabeled trials, which is standard cross-subject alignment practice
(analogous to Euclidean Alignment) and is low-risk specifically because:

- it touches no labels, so it cannot leak class information into the split;
- it only shifts/scales six power/entropy features by a mean and standard
  deviation computed over highly similar, class-balanced data (32 vs. 40
  trials from the same subject/session are not meaningfully different
  distributions).

It was nonetheless a real, literal deviation from "fit only on the training
portion" -- now closed (see the RESOLVED note above).

## Verdict

- **Subject-scope purity**: intact, enforced, and unit-tested. Not
  compromised.
- **Fold structure**: not an invented shortcut -- matches MOABB's own
  `WithinSessionEvaluation` defaults (5-fold, stratified, shuffled) for this
  specific single-session dataset.
- **Normalization scope**: was the one real deviation from a maximally
  strict protocol; now fits train-fold-only (see RESOLVED note above), so
  this is no longer a caveat.

My opinion: this is legitimate to use as a verification signal for the
dataset-generation pipeline, and with the normalization fix now in, there is
no remaining known compromise -- both the split mechanics (matching the
field's own standard tool) and the normalization scope (train-only, per
fold) hold up under a zero-compromise standard.

## Follow-up: does chance-level performance mean the graph data generation is broken?

After the normalization fix, a real 50-subject sweep
(`within_subject_full_v2`) confirmed the same population-level result as
before: balanced accuracy 0.522 [0.489, 0.556], AUC 0.532 [0.493, 0.571],
kappa 0.044 [-0.021, 0.111] -- all still centered on chance. But 8 of 50
subjects individually clear 0.65 balanced accuracy (7, 19, 20, 32, 36, 37,
40, 42), scattered across the full 1-50 ID range with no clustering. The
question this raised: does that pattern indict the graph data-generation
pipeline, or is it consistent with expected small-sample noise/real
between-subject variability?

### Checked so far

**Structural metadata (no anomalies found).** Every one of the 50 subjects
-- "good" and not -- has identical trial count (40), class balance (20/20),
sampling rate (500 Hz), trial duration (~2001 samples, ~4s, the ~0.1-0.3
sample jitter is normal per-trial epoching variance not a defect), and
consistent per-subject source file naming
(`sub-NN_task-motor-imagery_eeg.edf`). Rules out a batch/pipeline artifact
(e.g. a subset of subjects processed differently, mislabeled files, wrong
sampling rate) as the explanation for who ends up "good."

**A simple, graph-free classifier shows the "good" subjects carry real
signal, but the correspondence is imperfect.** Built the crudest possible
alternative representation per subject -- the 6 node features
(`without_csd`) mean-pooled across all 29 electrodes, plus the alpha-band
wPLI mean-pooled across all 812 edges (7 numbers per trial, no graph
structure, no neural net) -- and scored it with 5-fold stratified-CV LDA:

- The 8 GNN-"good" subjects average 0.612 balanced accuracy on this simple
  classifier vs. 0.484 for the other 42 -- a real, meaningful gap.
- Correlation between this simple classifier's per-subject score and the
  GNN's per-subject score across all 50 subjects: **r = 0.516**.
- Not a clean 1:1 relationship: subjects 32 and 36 are GNN-good (0.675
  each) but score *below chance* on the simple classifier (0.450, 0.425);
  subject 38 scores well on the simple classifier (0.625) but the GNN did
  poorly on it (0.425).

Interpretation: roughly a quarter of the variance in "which subjects the
GNN does well on" is explained by something visible even in trivial
mean-pooled band-power/connectivity, meaning the standout subjects are not
purely a fluke of one GNN training run. But the imperfect correspondence
means single-run GNN noise (only 8 held-out trials per fold) is also a real
contributor -- expected, not a bug, given the sample size.

### Not yet checked -- two open items before concluding "graph generation is fine"

1. **Edge/graph topology has not been isolated from node features.** The
   simple-classifier check above collapsed all 812 edges into one mean
   scalar -- it tests whether *some* connectivity signal exists, not
   whether `edge_index`/the per-edge wPLI values are structurally correct
   (right electrode pairs, no transposition/indexing bug, no duplicate or
   missing edges, values in wPLI's valid `[0, 1]` range). Proposed check:
   (a) a structural sanity pass over `edge_index` and the raw
   `wpli_without_csd` array (edge count == 29*28 == 812 confirmed already
   by `validate_dataset`, but not duplicate-pair or self-loop checks, nor a
   value-range check across all subjects/trials/bands); (b) an edges-only
   vs. nodes-only version of the simple-classifier check above (drop node
   features entirely and use only the 812 per-edge alpha wPLI values,
   flattened, with the same 5-fold LDA) to see whether connectivity alone
   carries separable signal independent of node power features, rather
   than only ever testing them combined.
2. **No comparison against Liu2024's own published baseline.** "Classic
   CSP+LDA motor imagery lands 65-80%" (cited earlier as an external
   reference point) is a rule of thumb from *other* motor-imagery datasets,
   not a confirmed number for Liu2024 itself. Chance-level average
   performance here could be the *expected* result for this specific
   dataset/task (some public MI datasets are known to be hard, with many
   BCI-illiterate subjects near chance) rather than a sign of anything
   broken in this pipeline. Proposed check: find Liu2024's original
   publication (or its MOABB dataset entry/documentation) and see what
   classification accuracy it reports, if any, as a same-dataset ceiling to
   compare against instead of a generic literature range.

Neither of these has been implemented yet -- recorded here so they aren't
lost, not decided unilaterally, since they involve either new analysis code
or external literature lookup rather than a small in-repo fix.
