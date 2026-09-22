# Liu2024 graph dataset plan

Based on [readmoabbliu.ipynb](readmoabbliu.ipynb). One sample represents one motor-imagery trial with 29 EEG channels as nodes.

## Generation approach and locations

Use **option two: generate and save the derived features**, then reuse them across experiments. Validate the calculation settings on a small subset before generating the full dataset. Experiments load the saved node and edge variants and select or combine them without repeating feature extraction.

| Location | Purpose |
| --- | --- |
| `data/moabb/MNE-liu2024-data/` | Existing source dataset, retained unchanged |
| `data/moabb/Graph-Liu2024-VersionTwo/` | Output directory to create for the saved feature dataset |
| `src/datautils/graphdataversiontwo/` | Python generation code, loader, notebook, and plan |

Save the calculation configuration and software versions with the dataset. If feature calculation settings change, regenerate the affected features in a separately identified dataset version. Fit normalization using training data when running experiments.

## Node features

Keep two node variants: calculate the same eight features from EEG **without CSD** and from EEG **after CSD**:

1. Delta band power (1–4 Hz).
2. Theta band power (4–8 Hz).
3. Alpha band power (8–13 Hz).
4. Beta band power (13–30 Hz).
5. Gamma band power (30–40 Hz).
6. Normalized spectral entropy (1–40 Hz).
7. Broadband Hjorth mobility (per sample).
8. Broadband Hjorth complexity (dimensionless).

Apply CSD to the time series before calculating the CSD-derived features. Both variants retain the same electrode nodes and feature order, with shape **`(29, 8)`** per trial.

Band powers are stored in dB with each variant's physical units and dB reference recorded in metadata; the implemented references are 1 µV² for EEG power and 1 (µV/m²)² for CSD power. Normalized spectral entropy is dimensionless in both variants.

## Edge features

Keep each connectivity method separately, calculated **with and without CSD**:

| Method | Without CSD | With CSD |
| --- | --- | --- |
| wPLI | Yes | Yes |
| PLV | Yes | Yes |
| Absolute imaginary coherence | Yes | Yes |

Each variant contains five bands: delta, theta, alpha, beta, and gamma. Use all 406 channel pairs, stored in both directions as 812 edges without self-loops. Edge-feature shape per variant per trial: **`(812, 5)`**.

All variants share the same subject, trial window, channel order, and edge order. Save the two node variants and six edge variants separately so later experiments can choose or combine node and edge features independently.

## Saved files

All files below belong in `data/moabb/Graph-Liu2024-VersionTwo/`. `N` is the number of saved trials.

| File | Contents |
| --- | --- |
| `node_features_without_csd.npy` | Six node features from EEG without CSD, `(N, 29, 8)` |
| `node_features_csd.npy` | Six node features from CSD-transformed EEG, `(N, 29, 8)` |
| `edge_index.npy` | Shared edge connections, `(2, 812)` |
| `edge_attr_wpli_without_csd.npy` | wPLI without CSD, `(N, 812, 5)` |
| `edge_attr_wpli_csd.npy` | wPLI with CSD, `(N, 812, 5)` |
| `edge_attr_plv_without_csd.npy` | PLV without CSD, `(N, 812, 5)` |
| `edge_attr_plv_csd.npy` | PLV with CSD, `(N, 812, 5)` |
| `edge_attr_icoh_abs_without_csd.npy` | Absolute iCoh without CSD, `(N, 812, 5)` |
| `edge_attr_icoh_abs_csd.npy` | Absolute iCoh with CSD, `(N, 812, 5)` |
| `labels.npy` | Left/right imagery target per trial, `(N,)` |
| `samples.tsv` | Subject, trial, source recording, and window boundaries in array order |
| `metadata.json` | Channel/feature order, signal source per variant, feature units and dB references, bands, label mapping, montage information, and calculation settings |
| `validation_report.json` | Saved-dataset verification results and subject/class counts |

## Python organization

Keep each task in a focused function with explicit inputs and returned outputs. Reuse the same feature functions for both signal variants, passing EEG or CSD data and the relevant settings. The notebook remains an inspection reference; generation must run from Python without notebook state.

Every function must include a descriptive docstring explaining:

- Its purpose and what it calculates or performs.
- Each input, including array shapes, units, and expected ordering where relevant.
- Its returned values, including shapes and units where relevant.
- Important assumptions, validation errors, and any input modifications or files written.

Use a consistent NumPy-style docstring format, with detail proportional to the function's complexity. Add inline comments where needed to explain non-obvious calculation choices.

The table below reflects the current implementation, so it can be checked against this plan directly
rather than only against the original intent. Private, underscore-prefixed helpers are omitted; every
row's public functions are what the plan's docstring/behavior requirements above apply to.

| Python file | Main functions and responsibilities |
| --- | --- |
| `config.py` | Keeps paths, band definitions, feature order, and calculation settings in one configuration, as planned |
| `source.py` | `discover_recordings()`, `load_recording()`, `read_trial_table()` — find local EDF files per subject, read local EEG, and identify trial boundaries, labels, and subject information. `discover_recordings()` was added beyond the original plan to support subject discovery in `generate_dataset.py` |
| `preprocessing.py` | `prepare_eeg()`, `attach_analysis_montage()`, `compute_csd()`, `extract_trial_pair()` — prepare EEG and CSD, then return matching trial windows, as planned |
| `node_features.py` | `compute_psd()`, `compute_band_powers()`, `compute_spectral_entropy()`, `build_node_features()` — produce the eight node features for either signal variant, as planned |
| `edge_features.py` | `build_edge_index()`, `compute_connectivity()`, `build_edge_features()` — calculate each connectivity method and map it to the shared edge order, as planned |
| `metadata.py` | `build_montage_metadata()`, `build_dataset_metadata()` — describe source coordinates, analysis geometry, feature meanings, and calculation settings, as planned. Also provides `file_sha256()` (shared checksum helper reused by `generate_dataset.py`) and `generation_settings()` (the fixed settings block embedded in dataset metadata), neither called out individually in the original plan |
| `validation.py` | `validate_trial_pair()`, `validate_features()`, `validate_saved_dataset()` — check alignment, dimensions, finite values, expected ranges, and saved-file consistency, as planned. Also exposes a `main()` CLI entry point (`python -m ...validation`) not mentioned in the original plan |
| `storage.py` | `save_dataset()` — write feature arrays, labels, sample records, and metadata, as planned |
| `generate_dataset.py` | `generate_subject_features()`, `main()` — coordinate the functions and generate all requested subjects, as planned. Also provides `calculation_fingerprint()`, which hashes settings/code/software to decide whether an interrupted run's checkpoints can be reused; this checkpoint-resume mechanism is implemented but not described in the plan |
| `saved_dataset.py` | `load_dataset()`, `get_graph()` — load saved arrays and expose selected node/edge combinations, as planned. `load_dataset()` returns a `SavedDataset` dataclass (memory-mapped arrays, samples, metadata) whose own `get_graph()` method is the one most callers use; this container type is not named in the plan |
| `__init__.py` | Package docstring only; no functions |

There is no `connectivity_cuda.py` in the current implementation. An earlier version of this package
included an optional float64 CUDA connectivity backend for faster generation; it was removed in favor of
generating exclusively through the CPU MNE reference implementation (`compute_connectivity()` in
`edge_features.py`), which every backend was always validated against, to keep this package's calculation
path to a single, directly auditable implementation.

### Tests

Not addressed by the original plan, but part of the current implementation:

| Python file | Coverage |
| --- | --- |
| `test_source.py` | Real local-data regression tests for marker decoding (subject 43's extra post-break code-2 pulses, subject 13's fractional marker tail) and CSD-vs-MNE-`sphere='auto'` parity. Skipped automatically when local Liu2024 files are unavailable |
| `test_connectivity_bands.py` | Verifies explicit inclusive-band averaging against a synthetic frequency ramp — the regression guard for the mne-connectivity 0.9.0 `_foi_average` band-doubling workaround described in `README.md` |
| `test_features.py` | Numerical checks on node/edge features: amplitude scaling versus dB, entropy behavior on a concentrated-power signal, phase-locked-signal connectivity, and reverse-edge symmetry |
| `test_dataset.py` | Synthetic-fixture tests for `saved_dataset`/`validation`: independent node/edge variant selection, copy isolation, and rejection of corrupted, mislabeled, or misaligned saved files |

Use one trial table and one channel/edge order for every variant. Check these at function boundaries, preserve the EEG input when computing CSD, and verify that saved arrays reload consistently. A failure must identify its subject/trial; do not silently omit a sample from only one variant.

The implementation and calculation choices are documented in [README.md](README.md). Source marker boundaries are preserved (2,000–2,002 samples per valid trial), and marker exceptions are recorded in metadata. The generated `validation_report.json` records the final saved-dataset checks.

## Broadband Hjorth node features

Each electrode in each trial now has eight features. Columns 0–4 are the original dB band powers, column 5 is normalized 1–40 Hz entropy, column 6 is Hjorth mobility, and column 7 is Hjorth complexity. Both EEG and CSD branches calculate their own descriptors from their marker-aligned trial signals without an additional band filter.

Hjorth uses population variances (`ddof=0`) of the signal and its first/second sample differences: mobility is `sqrt(v1/v0)` (per sample), complexity is `sqrt(v2/v1)/mobility` (dimensionless). Differences are not multiplied by sampling frequency. Undefined ratios and nonfinite inputs are rejected. Save unscaled values; fit any learned scaling only on training subjects.

See [NODE_FEATURE_EXPANSION_PLAN.md](NODE_FEATURE_EXPANSION_PLAN.md) for implementation and validation scope. Edge features remain five bands per method, with 812 stored entries including reverse pairs.
