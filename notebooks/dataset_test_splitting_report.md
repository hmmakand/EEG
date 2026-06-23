# Dataset Test Splitting Report

Scope: this report explains only `notebooks/dataset_test_splitting.ipynb`.

## Short Answer

The notebook uses a protocol-aware train/test split for `BNCI2014_001`, subject 3.
It does not randomly split the full dataset into train and test. Instead, it uses
the dataset metadata column `session`:

- `session == "0train"` becomes the training pool.
- `session == "1test"` becomes the final held-out test set.

This is the correct high-level idea for BCI Competition IV 2a style evaluation:
the official training session is used for model fitting and model selection, and
the official test session is kept separate for final evaluation.

## Dataset Loaded In The Notebook

The active dataset-loading cell is:

```python
subject_id = 3
dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[subject_id])
```

So the notebook is working with one subject only: subject 3 from `BNCI2014_001`.

The notebook output shows 12 recordings:

```text
12 recordings x 3 columns [subject, session, run]
```

Those 12 recordings are organized as:

```text
subject  session  run
3        0train   0
3        0train   1
3        0train   2
3        0train   3
3        0train   4
3        0train   5
3        1test    0
3        1test    1
3        1test    2
3        1test    3
3        1test    4
3        1test    5
```

This means the dataset already carries a natural split boundary. The split should
respect that boundary.

## Preprocessing Before Splitting

The active preprocessing cell does four things:

1. Keeps EEG channels.
2. Converts signal units from volts to microvolts.
3. Applies a 4-38 Hz band-pass filter.
4. Applies exponential moving standardization.

After preprocessing, the notebook creates event windows:

```python
windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)
```

The notebook uses:

```python
trial_start_offset_seconds = -0.5
sfreq = 250.0
trial_start_offset_samples = -125
```

The saved output shows each window has 1125 time samples. At 250 Hz, that is
4.5 seconds per window. Because no explicit `window_size_samples` or
`window_stride_samples` is provided, Braindecode creates one trial-aligned window
per event for this dataset.

## Window Counts

The notebook output shows:

```text
len(windows_dataset) = 576
metadata.shape = (576, 7)
```

The target labels are perfectly balanced:

```text
target  count
1       144
2       144
0       144
3       144
```

For `BNCI2014_001`, subject 3, this corresponds to:

- 288 windows from `0train`
- 288 windows from `1test`
- 576 total windows

So the train/test split is 50 percent / 50 percent by session, not because of a
random ratio, but because this dataset has equal-sized train and test sessions.

## Main Train/Test Split

The notebook performs the final train/test split here:

```python
splitted = windows_dataset.split("session")
train_set = splitted["0train"]
test_set = splitted["1test"]
```

This is the most important split in the notebook.

Conceptually:

```text
windows_dataset: 576 windows

0train session: 288 windows -> train_set
1test session:  288 windows -> test_set
```

The `test_set` should be treated as final held-out data. It should not be used
for hyperparameter tuning, early stopping decisions, model selection, or repeated
experimentation choices.

## First Training Example: No Validation Set

The first classifier uses:

```python
train_split=None
clf.fit(train_set, y=None)
test_acc = clf.score(test_set, y=y_test)
```

This means:

- The model trains on all 288 windows from `0train`.
- No validation set is used during training.
- The model is evaluated directly on all 288 windows from `1test`.

This is simple, but it is not ideal for model development because any repeated
changes guided by the test accuracy can slowly turn the final test set into an
informal validation set.

## Train/Validation/Test Split

The notebook later creates a validation split from the training session only:

```python
X_train = SliceDataset(train_set, idx=0)
y_train = np.array([y for y in SliceDataset(train_set, idx=1)])

train_indices, val_indices = train_test_split(
    X_train.indices_, test_size=0.2, shuffle=False
)

train_subset = Subset(train_set, train_indices)
val_subset = Subset(train_set, val_indices)
```

This is a better development setup because the validation data comes only from
`0train`. The final `1test` session remains untouched until evaluation.

The resulting counts are:

```text
train_set before validation split: 288 windows
train_subset:                     230 windows
val_subset:                        58 windows
test_set:                         288 windows
```

Why 58 validation windows? `test_size=0.2` means 20 percent of 288. That is
57.6, and scikit-learn rounds the test/validation side up to 58.

Because `shuffle=False`, the validation subset is the last 20 percent of the
training-session windows in their existing order. This avoids random mixing, but
it also means the validation set may correspond to later trials/runs rather than
a class-balanced random sample.

## How `predefined_split` Is Used

The validation example uses:

```python
train_split=predefined_split(val_subset)
clf.fit(train_subset, y=None)
```

This tells Skorch/Braindecode:

- Fit model weights using `train_subset`.
- Evaluate validation metrics using `val_subset`.
- Do not draw validation data from `test_set`.

This is the correct relationship between training, validation, and final testing.

## K-Fold Validation

The notebook also demonstrates K-fold cross-validation:

```python
train_val_split = KFold(n_splits=5, shuffle=False)
cv_results = cross_val_score(
    clf, X_train, y_train, scoring="accuracy", cv=train_val_split, n_jobs=1
)
```

Important detail: K-fold is applied to `X_train` and `y_train`, which come from
`train_set`, not from `windows_dataset`.

So the K-fold procedure divides only the `0train` session into folds. The
`1test` session remains outside cross-validation.

For 288 training windows and 5 folds, the validation fold sizes are approximately:

```text
fold 1: 58 validation windows
fold 2: 58 validation windows
fold 3: 58 validation windows
fold 4: 57 validation windows
fold 5: 57 validation windows
```

The saved notebook output reports:

```text
Validation accuracy: 29.51+-6.62%
```

That value is a validation result from the training session, not a final test
result.

## Grid Search Split

The grid search cell defines:

```python
train_val_split = [
    tuple(train_test_split(X_train.indices_, test_size=0.2, shuffle=False))
]
```

Then:

```python
search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=train_val_split,
    return_train_score=True,
    scoring="accuracy",
    refit=True,
)

search.fit(X_train, y_train)
```

This again uses only the `0train` session, because `X_train` and `y_train` were
built from `train_set`.

The grid search compares learning rates using a fixed validation split from the
training session. It does not use the final `1test` session during the search.

One caveat: after `GridSearchCV(..., refit=True)`, the best estimator is refit by
scikit-learn on all data passed to `search.fit`, meaning all 288 windows from
`0train`. That is acceptable if final evaluation is done once on `test_set`
after model selection.

## Important Caveats

### 1. The notebook has stale output in a commented cell

Cell 1 is commented out, but it still has saved output from an earlier Liu2024
run. That output is not part of the active BNCI2014_001 split logic. It can
confuse readers because the code is commented while output remains visible.

### 2. Preprocessing happens before the session split

The notebook preprocesses the full subject dataset before splitting into
`0train` and `1test`.

Filtering each raw recording is generally fine because recordings are separate.
However, any preprocessing step that estimates statistics from the complete
dataset can create leakage if it learns from both train and test together.

In this notebook, exponential moving standardization is applied per continuous
recording, so the leakage risk is lower than a global scaler fitted across all
windows. Still, for strict experimental hygiene, the safest pattern is:

1. Load raw data.
2. Apply recording-local signal preprocessing.
3. Window the data.
4. Split by `session`.
5. Fit any learned transforms only on the training side.

### 3. `shuffle=False` preserves order

The validation and K-fold splits use `shuffle=False`. This is often reasonable
for EEG windows because neighboring windows/trials can be correlated. But it
also means validation folds are contiguous blocks. If the trial order has
session/run/class structure, validation scores can depend on that ordering.

### 4. Final test accuracy should not guide development

The notebook prints test accuracy multiple times. That is useful for learning
the API, but in a real experiment the final test set should be checked only
after preprocessing, model family, hyperparameters, and training settings are
chosen using training/validation data.

## Recommended Interpretation

The intended split hierarchy in this notebook is:

```text
BNCI2014_001 subject 3
`-- windows_dataset: 576 windows
    |-- train_set: 0train session, 288 windows
    |   |-- train_subset: 230 windows
    |   `-- val_subset: 58 windows
    `-- test_set: 1test session, 288 windows
```

For K-fold validation:

```text
BNCI2014_001 subject 3
`-- windows_dataset: 576 windows
    |-- train_set: 0train session, 288 windows
    |   `-- 5-fold cross-validation happens here only
    `-- test_set: 1test session, 288 windows
```

## Bottom Line

The notebook's core splitting logic is sound:

- It uses the official/session metadata split for final train/test separation.
- It creates validation data only from the training session.
- K-fold and grid search are also restricted to the training session.

The main thing to be careful about is experimental discipline: use validation
results for model choices, and reserve `test_set` for the final evaluation.
