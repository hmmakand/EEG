# Plan: eight node features with broadband Hjorth descriptors

Status: implemented and pilot-validated. The specification and implementation steps below are retained as the design record; the implementation now uses schema 2.

## Implementation results

- Both EEG and CSD produce eight named columns; storage, loader, metadata, validation, and notebook representations have been updated.
- Existing suite: **20 tests passed, 24 subtests passed**. One existing MNE montage-name deprecation warning remains.
- Generated and validated `data/moabb/Graph-Liu2024-VersionTwo-hjorth-pilot`: subjects 1 and 43, two trials each, two left-hand and two right-hand labels overall. Both node arrays have shape `(4, 29, 8)`; all six edge variants retain `(4, 812, 5)`.
- Recomputed a real EEG/CSD trial independently of storage: the first six columns match version one's builder exactly and all eight float32 columns match the saved pilot.
- Changing the schema setting changes the calculation fingerprint; resume checks require an exact fingerprint/signature match.
- Executed the main eight-feature and alternative four-feature notebook cells and their stacked plots on real EEG. Main notebook values match the shared builder.
- Executed the saved-dataset notebook's relevant inspection, graph, validation, and plotting cells against the pilot. Full-dataset balance cells were excluded because they require 50 subjects and 2,000 trials.
- Full dataset generation and full-notebook execution remain outside this completed pilot validation. Version-one code and data remain unchanged. No alpha-only edge change was implemented.

## Objective and unit of calculation

Extend the current six node features with Hjorth mobility and Hjorth complexity. Build one graph for each subject's individual motor-imagery trial, labeled left hand (0) or right hand (1). Calculate all node features separately for each electrode within that trial's existing marker-aligned window.

“Broadband” here means the trial signal without additional frequency-band filtering. It does not mean combining electrodes, trials, or subjects, and it does not imply a new 1–40 Hz filter. Existing source preprocessing is retained.

Apply the same feature definitions independently to both node variants:

- `without_csd`: the EEG trial in volts.
- `csd`: the corresponding CSD trial in volts per square metre.

The CSD variant must calculate Hjorth from its own CSD signal, not copy the EEG Hjorth values.

## Feature contract

| Column | Name | Definition / units |
| --- | --- | --- |
| 0 | `delta_power_db` | Existing 1–4 Hz power in dB |
| 1 | `theta_power_db` | Existing 4–8 Hz power in dB |
| 2 | `alpha_power_db` | Existing 8–13 Hz power in dB |
| 3 | `beta_power_db` | Existing 13–30 Hz power in dB |
| 4 | `gamma_power_db` | Existing 30–40 Hz power in dB |
| 5 | `spectral_entropy` | Existing normalized 1–40 Hz spectral entropy, dimensionless, [0, 1] |
| 6 | `hjorth_mobility` | Broadband temporal mobility, per sample |
| 7 | `hjorth_complexity` | Broadband temporal complexity, dimensionless |

Keep the first six calculations and their order unchanged, including the existing EEG/CSD dB references. Per-trial matrices become `(29, 8)`; each saved node array becomes `(N, 29, 8)` in float32. Compute features in float64 before storage conversion.

Keep edge features, channel order, trial boundaries, labels, and subject identifiers unchanged. The notebook's separate four-feature/selected-band experiment remains an illustrative alternative, not the generation policy.

## Hjorth definition

Match the appended notebook's discrete sample-difference convention. For each channel's trial `x`:

```python
d1 = np.diff(x)
d2 = np.diff(x, n=2)
v0 = np.var(x, ddof=0)
v1 = np.var(d1, ddof=0)
v2 = np.var(d2, ddof=0)
mobility = np.sqrt(v1 / v0)
complexity = np.sqrt(v2 / v1) / mobility
```

Differences use only samples within the trial and are not multiplied by sampling frequency. Document this convention and the recording sampling rate in metadata. Mobility values using this convention depend on sampling rate; do not silently mix recordings at different rates.

Require a finite, two-dimensional channel-by-sample signal with at least three samples. Reject zero signal variance or zero first-difference variance with an informative error; do not replace undefined ratios with zero or an arbitrary epsilon. Check final results for finiteness. Preserve input arrays. Include subject/trial context in generation failures and identify affected channels where possible.

Mobility must be positive for accepted inputs; complexity must be nonnegative. Neither feature is a normalized [0, 1] feature. Do not impose a complexity >= 1 rule on finite-window estimates.

## Implementation steps

1. **Configuration and numerical calculation**
   - Append the two names to `config.NODE_FEATURE_NAMES`.
   - Add a documented `compute_hjorth_features(trial)` function in `node_features.py`, returning `(n_channels, 2)` in the specified order.
   - Append its output in `build_node_features()` after entropy.
   - Confirm `generate_dataset.py` applies this builder to both aligned signal branches and propagates useful failure context.

2. **Storage, loading, and validation**
   - Update `storage.py` node widths from six to eight, preferably deriving widths from the declared feature contract.
   - Update the shape/name contract in `saved_dataset.py`, including returned graph documentation and optional PyG output.
   - Update `validation.py` to validate eight columns. Entropy stays at index 5: remove the current assumption that it is the final column, locating it by the declared feature name.
   - Check Hjorth finiteness and valid signs separately from entropy's [0, 1] bounds, both before storage and when validating saved arrays.
   - Validate exact feature names/order so mislabeled eight-column arrays cannot pass just by having the correct width.
   - Search all version-two Python files and notebooks for six-column shapes and entropy-last indexing; update each relevant assumption.

3. **Schema and provenance**
   - Set version two's `SCHEMA_VERSION` to 2 when implementing the eight-feature contract. The current copy uses schema 1 with six features.
   - Make version two's loader reject unsupported schema versions with an actionable error. Existing six-feature datasets remain loadable through version one's loader; do not silently pad or reinterpret them.
   - Add Hjorth formula, difference order, variance convention, input signal scope, units, and invalid-input policy to `metadata.py` generation settings and node-unit descriptions.
   - Retain the generation fingerprint so changed code/settings invalidate old checkpoints. Verify this behavior before using resume.

4. **Notebook and documentation**
   - Update the main node-feature demonstration in `readmoabbliu.ipynb` to show the eight-feature representation and stacked profiles, with correct labels and scales.
   - Keep the separate four-feature section clearly labeled as an alternative. Prefer the shared helper for Hjorth to avoid implementation drift.
   - Update `test_generated_dataset.ipynb` shape checks, entropy indexing, descriptions, and feature plots for the eight-feature dataset.
   - Update `README.md` and `DATASET_PLAN.md` to describe the implemented contract only once implementation is complete. Clear obsolete execution outputs before rerunning affected notebook cells.

## Verification and acceptance criteria

- Numerical tests establish Hjorth amplitude-scale and constant-offset invariance, without altering inputs.
- A sampled sinusoid provides an independent numerical reference: compare mobility with the sample-difference expectation and complexity near one using suitable finite-window tolerances.
- Reject constant signals, signals with zero first-difference variance, insufficient samples, and nonfinite inputs explicitly.
- Verify the first six outputs match the pre-expansion calculations on the same signal.
- Exercise EEG and CSD separately on a real marker-aligned trial; both must yield finite `(29, 8)` matrices from their respective inputs.
- Update existing dataset fixtures and test saved-file round trips, graph copy isolation, incorrect width/order, invalid Hjorth values, entropy corruption specifically at column 5, and unsupported schema rejection.
- Run the existing version-two test suite after the focused checks.
- Generate a small pilot in `data/moabb/Graph-Liu2024-VersionTwo-hjorth-pilot` using subjects 1 and 43, two trials each. Validate both node variants, all unchanged edge variants, labels, provenance, and schema after reloading.
- Execute the affected notebook calculation and plotting cells against real data. Record what was executed; do not claim full-notebook or full-dataset verification from a pilot.

## Data generation and scaling

After successful implementation and pilot validation, full generation can target `data/moabb/Graph-Liu2024-VersionTwo`. If that directory already exists, use a new explicit output path; preserve existing datasets. Never overwrite or modify version one's files. Full dataset generation is a later execution step, not part of writing this plan.

Save dB powers and unscaled Hjorth values. Any learned scaling belongs in the training workflow, fitted only on training subjects within each split/fold and applied unchanged to held-out subjects. Fit EEG and CSD scaling separately. Entropy keeps its current normalization.

Completion means both node variants generate, save, load, and validate eight correctly named features per electrode per trial, with unchanged graph labels and edge representations, and the focused tests and pilot passing.
