# Liu2024 graph features

This package generates two node variants and six edge variants from the existing local Liu2024 EDF files. Each sample has 29 electrodes, six node features, and five values per edge per connectivity variant. It reads local files directly through MNE and uses the notebook calculations.

## Generate and verify

Run from the repository root with the numerical dependencies in this directory's `requirements.txt` installed. The current environment already has them.

```bash
# Small pilot in a separate directory.
python -m src.datautils.graphdataversionone.generate_dataset \
  --subjects 1 43 --limit-trials 2 --workers 2 \
  --output data/moabb/Graph-Liu2024-VersionOne-pilot

# All 50 subjects using the MNE reference implementation.
# Output: data/moabb/Graph-Liu2024-VersionOne/
python -m src.datautils.graphdataversionone.generate_dataset --workers 4

# Read every saved feature and check consistency again.
python -m src.datautils.graphdataversionone.validation \
  data/moabb/Graph-Liu2024-VersionOne
```

Generation rejects an existing output directory. Interrupted runs keep subject checkpoints in a sibling hidden `.working` directory; rerunning the same command reuses only checkpoints whose source files, code, settings, software, and contents match. A completed dataset is published only after all variants pass validation. Calculation changes require a new output directory/version.

## Load combinations

```python
from src.datautils.graphdataversionone.saved_dataset import load_dataset

dataset = load_dataset("data/moabb/Graph-Liu2024-VersionOne")
graph = dataset.get_graph(
    0,
    node_variant="csd",
    edge_variants=("wpli_without_csd", "plv_csd"),
)
print(graph["x"].shape)          # (29, 6)
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

## Generated dataset verification

The completed `data/moabb/Graph-Liu2024-VersionOne/` dataset contains 2,000 trials from all 50 subjects (40 each; 20 left-hand and 20 right-hand). Both node variants and all six edge variants passed saved-file validation. Details are in the output directory's `validation_report.json`.

The package's 16 tests pass, and the updated notebook executes from a fresh namespace. The completed dataset occupies approximately 189 MiB.
