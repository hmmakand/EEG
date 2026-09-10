# Liu2024 preprocessing audit

Audit date: 2026-09-03

## Scope

This audit is limited to `src/datautils/MoabbLiu2024`. It covers the Python
pipeline, notebook, tests, saved-array interface, and the saved output produced
by this package. It does not audit later graph generation, feature extraction,
models, or training.

Audit verification:

- `PYTHONPATH=/workspaces/EEG pytest -q src/datautils/MoabbLiu2024/test_liu2024.py`
  passed: **5 tests passed** with third-party warnings.
- The saved 31--40 Hz dataset was inspected directly. All four arrays are
  finite and have the declared shapes and dtypes.
- Every saved subject has exactly 40 trials, trial IDs 0--39, and a 20/20
  left/right class balance.

## What this package produces

The package transforms locally cached Liu2024 EDF recordings into a dense,
memory-mappable motor-imagery dataset. Its current gamma-band output is:

```text
data/moabb/Preprocessed-Gamma-31-40-MNE-liu2024-data/
  X.npy
  y.npy
  subject_ids.npy
  trial_indices.npy
  metadata.json
```

The observed complete dataset contains:

| Array | Shape | Type | Meaning |
|---|---:|---|---|
| `X.npy` | `(2000, 29, 2000)` | `float32` | EEG trial windows |
| `y.npy` | `(2000,)` | `int64` | 0=left hand, 1=right hand |
| `subject_ids.npy` | `(2000,)` | `int64` | subject IDs 1--50 |
| `trial_indices.npy` | `(2000,)` | `int64` | within-subject window index 0--39 |

There are 50 subjects and 40 examples per subject. Each subject contributes 20
left-hand and 20 right-hand trials, giving 1,000 examples per class. A single
sample is 29 channels by 2,000 samples at 500 Hz: a four-second EEG window.

The stored channel order is:

```text
FP1, FP2, Fz, F3, F4, F7, F8, FCz, FC3, FC4, FT7, FT8,
Cz, C3, C4, T3, T4, CP3, CP4, TP7, TP8, Pz, P3, P4,
T5, T6, Oz, O1, O2
```

## Processing flow

```text
local Liu2024 EDF
  -> MOABB/Braindecode recording objects
  -> validate annotations and channel consistency
  -> retain EEG channels only
  -> 31--40 Hz band-pass filter
  -> resample to 500 Hz
  -> exponential moving standardization
  -> extract left/right event windows
  -> convert to float32 NumPy arrays
  -> validate aggregate counts and finite values
  -> atomically publish the saved directory
```

### 1. Local loading

`loader.py` accepts subject IDs 1--50, rejects empty/duplicate/invalid lists,
and checks that every requested EDF file exists under the local MOABB cache.
It sets `MNE_DATASETS_LIU2024_PATH` and constructs a Braindecode
`MOABBDataset`. This intentionally prevents a silent download, but setting the
environment variable is a process-wide side effect.

The loader delegates EDF parsing, annotation conversion, physical-unit
handling, and recording construction to the installed MOABB/MNE/Braindecode
versions.

### 2. Raw validation

`validation.py` checks that:

- at least one recording exists;
- every recording contains both `left_hand` and `right_hand` annotations;
- every recording has EEG channels;
- EEG names and ordering are identical across recordings;
- after channel selection, only EEG remains;
- after preprocessing, sampling frequency is 500 Hz.

It also returns recording-level subject/session/run information and event
counts. It does not require exactly 20 annotations per class at this stage.

### 3. Channel selection

`channels.py` applies MNE's `pick(picks="eeg")` operation in place. The local
data test confirms that this leaves the expected 29 EEG channels. No spatial
reordering or interpolation is performed: the order supplied by the source
recording is retained and checked for consistency.

### 4. Signal preprocessing

`preprocessing.py` mutates each continuous recording in this order:

1. MNE band-pass filtering at 31--40 Hz;
2. resampling to 500 Hz;
3. optional Braindecode exponential moving standardization.

The current standardization uses `factor_new=0.001` and an initialization block
of 2,000 samples (four seconds at 500 Hz). Importantly, standardization occurs
on the continuous recording **before event windows are extracted**. It is
therefore a running, recording-level transformation, not independent
normalization of each four-second trial. Earlier signal history can influence
the standardized values in a later trial.

This stage does not create handcrafted or topological features. It only
produces filtered, resampled, standardized EEG time series.

### 5. Event windowing

`windowing.py` calls Braindecode's `create_windows_from_events` and maps:

```text
left_hand  -> 0
right_hand -> 1
```

Both configured offsets default to zero, so the complete annotated
four-second imagery interval is used. At 500 Hz this yields `(29, 2000)` per
trial. Windowing occurs after continuous preprocessing. The returned third
item is Braindecode window metadata; the saver discards it and creates its own
within-subject sequential index.

### 6. Saving

`save_preprocessed_dataset.py` processes subjects one at a time to constrain
memory use. It preallocates `.npy` memory maps, casts EEG to `float32`, stores
integer targets/provenance as `int64`, and writes metadata. Generation happens
in a temporary sibling directory and is published with `os.replace`; an
existing destination is protected unless `overwrite=True`.

When overwrite is enabled, the existing output directory is removed immediately
before publication. The temporary-generation strategy protects against most
partial outputs, although replacement is not a rollback transaction after the
old directory has been removed.

### 7. Saved-data access

`torch_moabbliu2024.py` memory-maps the four arrays. `__getitem__` copies only
the requested window into a writable `float32` PyTorch tensor and returns:

```text
(signal_tensor, integer_target, {subject, trial})
```

This is efficient: the complete approximately 464 MB signal array does not
need to be loaded into heap memory.

## Findings

### High: saver and PyTorch loader defaults point to different datasets

The saver defaults to:

```text
Preprocessed-Gamma-31-40-MNE-liu2024-data
```

but `Liu2024TorchDataset()` defaults to:

```text
Preprocessed-MNE-liu2024-data
```

The notebook constructs `Liu2024TorchDataset()` without a path, so its saved
dataset inspection can silently examine the older/default dataset rather than
the gamma output produced by the current saver. This can cause later code to
train on the wrong frequency band while appearing to use this pipeline.

Recommendation: define one canonical preprocessed-output constant in
`config.py` and import it in both saver and loader. Require metadata band
validation when the gamma dataset is expected.

### Medium: saved-data validation is too shallow

`Liu2024TorchDataset._validate()` checks array lengths, the rank of `X`, and
two metadata dimensions. It does not check declared dtypes, finite EEG values,
valid labels, subject range, channel count, sampling frequency, per-subject
balance, unique trial IDs, or consistency between metadata subjects and saved
IDs. Corrupted or incorrectly selected data can therefore load successfully.

Recommendation: add a strict saved-dataset validator and call it by default or
offer an explicit strict mode. Validate the complete fixed contract for the
canonical dataset.

### Medium: generation checks only aggregate class balance

The saver requires 40 windows per subject and 1,000 examples per class across
the complete dataset, but does not require 20 examples of each class for every
subject. Opposite per-subject imbalances could cancel at the aggregate level.
Direct inspection confirms the current saved arrays are correctly balanced,
but the invariant is not enforced by generation code.

Recommendation: validate per-subject class counts and trial-index uniqueness
before publishing.

### Medium: preprocessing provenance is incomplete

Metadata records frequency bounds, resampling frequency, standardization
parameters, offsets, channel names, and basic formats. It does not record:

- MOABB, MNE, Braindecode, NumPy, or SciPy versions;
- exact filter design, phase, transition bandwidth, padding, or resampling
  implementation/settings;
- source EDF hashes or source dataset revision;
- generation command, timestamp, Git revision, or dirty state;
- explicit duration/event-boundary semantics.

Defaults in third-party libraries can change, so two runs with the same local
metadata are not guaranteed to be identical.

Recommendation: record resolved preprocessing parameters, package versions,
source hashes, and code revision in `metadata.json`.

### Medium: continuous standardization needs an explicit scientific rationale

Exponential moving standardization is applied before trials are cut. This may
be intentional and is causal with respect to signal order, but every window is
conditioned on earlier portions of the same continuous recording. It is not
equivalent to trial-wise z-scoring or normalization fitted on training
subjects. It may also make results sensitive to recording order and preceding
rest/task intervals.

Recommendation: describe this exact scope in the methods. Compare continuous
EMA standardization with no standardization and a clearly leakage-safe
alternative. Do not describe the current operation merely as “normalized”
without its temporal scope.

### Low: raw validation checks event presence rather than exact protocol

Before window creation, validation only requires both target event names to be
present. It does not reject unexpected task annotations, enforce exactly 20
left and 20 right events, validate four-second annotation durations, or verify
subject/session/run identities against the requested configuration.

Recommendation: validate protocol-level counts, durations, recording identity,
and permissible annotations before preprocessing.

### Low: configuration validation is incomplete

`factor_new` is not checked to be finite and within its meaningful range, and
the standalone preprocessing function does not validate `init_block_size` or
`n_jobs`. Event mapping keys are checked but distinct integer values are not
validated at the windowing boundary. The save function also accepts a subject
tuple directly and validates some errors only later through per-subject
configuration construction.

Recommendation: validate all public-function parameters at entry and require
the class mapping to be exactly the intended binary mapping unless alternate
labels are deliberately supported.

### Low: tests cover the normal path but not saved artifacts

The five tests cover configuration, local loading, preprocessing, channel
count, window shape, finiteness, and class balance for subject 1. There are no
tests for atomic saving, overwrite behavior, memory-mapped loading, malformed
metadata/arrays, the default-path mismatch, all-subject invariants, or exact
preprocessing provenance.

Recommendation: add temporary-directory unit tests for saving/loading and a
fast strict-validation test against small synthetic arrays.

## Overall assessment

The processing order is clear, modular, and operational, and the observed
gamma dataset has the intended shape, channel order, class balance, trial IDs,
and finite values. The main correctness risk is the inconsistent default path
between generation and loading. The main scientific reproducibility risks are
under-specified third-party preprocessing and the fact that exponential moving
standardization is continuous-recording based rather than trial-local. Fixing
the path contract and strengthening saved-artifact validation should precede
further downstream experiments.
