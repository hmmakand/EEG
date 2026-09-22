# Liu2024 graph features — version two

This package generates two node variants and six edge variants from the existing local Liu2024 EDF files. Each sample has 29 electrodes, eight node features, and five values per edge per connectivity variant. It reads local files directly through MNE and uses the notebook calculations.

## Generate and verify

Run from the repository root with the numerical dependencies in this directory's `requirements.txt` installed. Install notebook dependencies (`matplotlib`, `networkx`, and `ipython`) separately when using the inspection notebooks.

```bash
# Small pilot in a separate directory.
python -m src.datautils.graphdataversiontwo.generate_dataset \
  --subjects 1 43 --limit-trials 2 --workers 2 \
  --output data/moabb/Graph-Liu2024-VersionTwo-pilot

# All 50 subjects using the MNE reference implementation.
# Output: data/moabb/Graph-Liu2024-VersionTwo/
python -m src.datautils.graphdataversiontwo.generate_dataset --workers 12

# Read every saved feature and check consistency again.
python -m src.datautils.graphdataversiontwo.validation \
  data/moabb/Graph-Liu2024-VersionTwo
```

Generation rejects an existing output directory. Interrupted runs keep subject checkpoints in a sibling hidden `.working` directory; rerunning the same command reuses only checkpoints whose source files, code, settings, software, and contents match. A completed dataset is published only after all variants pass validation. Calculation changes require a new output directory/version.

## Load combinations

```python
from src.datautils.graphdataversiontwo.saved_dataset import load_dataset

dataset = load_dataset("data/moabb/Graph-Liu2024-VersionTwo")
graph = dataset.get_graph(
    0,
    node_variant="csd",
    edge_variants=("wpli_without_csd", "plv_csd"),
)
print(graph["x"].shape)          # (29, 8)
print(graph["edge_index"].shape) # (2, 812)
print(graph["edge_attr"].shape)  # (812, 10)

# Optional: directly obtain torch_geometric.data.Data.
pyg_graph = dataset.get_graph(0, node_variant="without_csd",
                              edge_variants=("wpli_csd",), as_pyg=True)
```

Arrays are memory mapped read-only. Returned graphs contain copies so experiment code can transform them without modifying the dataset. Fit any learned scaling or feature selection using training data only; subject IDs and trial records support subject-based splits.

## Calculation choices

- EEG uses the 29 channels inspected in the notebook. There is no additional temporal filtering, resampling, re-referencing, artifact rejection, or normalization in this generator. Source EDF processing is retained.
- CSD is MNE's spherical spline surface Laplacian on a copied signal with the notebook's `standard_1020` analysis template, fitted sphere, regularization `1e-5`, stiffness `4`, and `50` Legendre terms. The original electrode TSV is preserved separately because its units/frame are undeclared.
- Nodes use two-second Welch segments with 50% overlap, Simpson-integrated band powers, and normalized 1–40 Hz spectral entropy. Bands are delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–40 Hz, including both boundaries as in the notebook.
- EEG powers use dB relative to `1 µV²`; CSD powers use dB relative to `1 (µV/m²)²`. Entropy and connectivity are dimensionless. This distinction is saved in metadata.
- Connectivity uses the notebook's single-trial multitaper `spectral_connectivity_time` settings. The three methods share a spectral calculation. Band averaging is performed explicitly: the installed mne-connectivity 0.9.0 helper incorrectly doubles upper frequency indices for multi-bin bands. This is a correction to earlier notebook edge outputs; the notebook now uses the same explicit bounds. Absolute iCoh is applied after signed band averaging. All 406 pairs are mirrored to 812 directed storage entries, with no threshold or self-loops.
- Marker decoding verifies the 40 instruction anchors against the source protocol and selects one unambiguous code-2/code-3 pair per trial block. Observed durations are 2,000–2,002 samples; their actual boundaries are preserved for both signal branches. Fractional marker-channel fill and extra pulses after a block's MI endpoint are recorded in each subject's `marker_audit`. In particular, two extra code-2 pulses in subject 43 are not assigned trial labels.
- Trial classes come from the supplied events table (`trial_type=1`: left hand, saved label `0`; `trial_type=2`: right hand, label `1`). Actual EDF markers determine boundaries. A single trial table aligns every node/edge variant.

Single-trial connectivity is an exploratory estimate, especially at low frequencies within roughly four-second windows. Saving a verified dataset establishes reproducible calculations, not predictive validity of its features.

The concise feature/file specification remains in [DATASET_PLAN.md](DATASET_PLAN.md). Every implementation function documents its purpose, inputs, outputs, units, and relevant assumptions.

## Version-two isolation and verification

Version two uses its own Python modules, output directory (`data/moabb/Graph-Liu2024-VersionTwo/`), and sibling checkpoint directory (`.Graph-Liu2024-VersionTwo.working/`). It shares the original local `MNE-liu2024-data` input. Paths assume this repository layout; generation accepts `--source-root` and `--output` overrides.

The saved-file schema is version 2: five dB band powers, normalized spectral entropy, Hjorth mobility, and Hjorth complexity, in that order. Version two rejects schema-1 files; use version one to load the original six-feature datasets.

A four-trial schema-2 pilot for subjects 1 and 43 is available in `data/moabb/Graph-Liu2024-VersionTwo-hjorth-pilot/`; both node variants and all edge variants passed saved-file validation. A full version-two dataset has not been generated. After generation, use the validation command above and `test_generated_dataset.ipynb` to inspect version-two results. Copied notebook outputs have been cleared; the historical audit in `verdict_temp.md` describes version one only.

Run the package tests from the repository root:

```bash
python -m pytest src/datautils/graphdataversiontwo/
```

## Broadband Hjorth node features

Each electrode in each trial now has eight features. Columns 0–4 are the original dB band powers, column 5 is normalized 1–40 Hz entropy, column 6 is Hjorth mobility, and column 7 is Hjorth complexity. Both EEG and CSD branches calculate their own descriptors from their marker-aligned trial signals without an additional band filter.

Hjorth uses population variances (`ddof=0`) of the signal and its first/second sample differences: mobility is `sqrt(v1/v0)` (per sample), complexity is `sqrt(v2/v1)/mobility` (dimensionless). Differences are not multiplied by sampling frequency. Undefined ratios and nonfinite inputs are rejected. Save unscaled values; fit any learned scaling only on training subjects.

See [NODE_FEATURE_EXPANSION_PLAN.md](NODE_FEATURE_EXPANSION_PLAN.md) for implementation and validation scope. Edge features remain five bands per method, with 812 stored entries including reverse pairs.
