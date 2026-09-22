# Original EEG partitions

This package extracts original trial segments and makes fixed, within-subject
training/validation/testing partitions. It does not create synthetic trials,
augmentations, filtered signals, features, or graphs. It reuses `dataset/`'s
MATLAB loading and trial extraction functions, not `build_subject_dataset()`.

From the repository root:

```bash
source .venv/bin/activate
python -m src.ourexperimentversionthirteen.datasetsynthetic.original \
  --data-dir data/kaggle/bci-competition-iv-data-sets-2a \
  --subjects 1 2 3 5 6 7 8 9 \
  --seed 42
```

From a standalone copy of version thirteen:

```bash
python datasetsynthetic/original.py --data-dir /path/to/data --seed 42
```

The default output is `datasyn/original/` inside version thirteen. Use `--output`
to create a separate dataset variant. Existing output is verified and reused;
a different configuration, changed input files, or changed saved artifacts causes
an error instead of silently replacing partitions.

## Rules

Settings live in `config.py`. Defaults extract four-second segments at 250 Hz,
using session indices 3–8 and channel indices 0–21. Source labels 1/2 map to
left/right targets 0/1. The event-index offset remains zero, matching the existing
loader; no automatic MATLAB one-based correction is applied. Incomplete trials
are excluded and recorded. Unselected sessions are specified by configuration;
unselected classes within selected sessions are recorded as exclusions.

Each class is shuffled independently per subject, then split 70/15/15. Counts use
integer largest-remainder rounding; ties prioritize training, validation, testing.
Thus 60 trials per class yields 42/9/9; 72 yields 50/11/11. Subjects unable to
populate both classes in all three partitions are rejected. Every eligible trial
is retained exactly once, including odd trial totals. Legacy interleaving is not
used. Each completed partition is shuffled again.

One fixed seed drives independent PCG64 streams through
`SeedSequence([seed, subject_id, stage, class_or_partition_index])`, with stage 0
for class shuffling and stage 1 for final partition shuffling. Processing one
subject alone does not change its partitions. Saved assignments and order are
authoritative for subsequent experiments.

## Saved files

- `subject_XX.npz`: `signals` shaped `[trials, 1000, 22]`, `labels`, `subject_ids`,
  `session_ids`, and `original_trial_indices`. Original sample values and dtype
  are preserved. Rows follow original session/event order.
- `partitions.csv`: source filename, subject/session IDs, original event index,
  original event position, extracted start sample, original and mapped labels,
  array row index, assigned partition, order within that partition, and split seed.
- `metadata.json`: extraction/split settings, source and artifact SHA-256 hashes,
  NumPy version, randomization and rounding rules, per-class counts, exclusions,
  and index conventions. Session/event/array/partition indices are zero-based.

A stable trial identity is `(source_file, session_id, original_trial_index)`.
These are original extracted EEG segments, not a claim that the supplied source
recordings had never undergone preprocessing.

## Load the same partitions in every experiment

```python
from src.ourexperimentversionthirteen.datasetsynthetic.original import load_partition

signals, labels, trial_records = load_partition(
    'src/ourexperimentversionthirteen/datasyn/original',
    subject_id=1,
    partition='training',  # or 'validation' / 'testing'
)
```

The loader verifies saved artifacts and returns samples in saved partition order.
Do not resplit these trials for individual models. Fit normalization and other
learned preprocessing on training only; augment training only. Select epochs
using validation, then evaluate the selected model on testing. Sessions may
appear in multiple partitions: this is trial-level within-subject evaluation.

The existing supervised KFold trainer is unchanged and does not consume this
new dataset automatically. Its scores are a separate evaluation protocol.
