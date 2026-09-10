# Liu2024 manuscript broadcast-11 experiment

This package is dedicated exclusively to the saved dataset:

```text
data/moabb/Graph-Gamma-PLV-Manuscript-30-40-11-liu2024-data
```

Generated from a dedicated 31-40 Hz gamma-band source (matching the
manuscript's stated gamma band), distinct from the 31-80 Hz source used by
`current_v1`/`manuscript_candidate_v2`. See
`src/datautils/PlvLiu2024Broadcast11/DATASET_GENERATION.md`.

Every sample contains 29 EEG-channel nodes with 11 node-feature columns:

```text
MAV, STD, PSD, betweenness,
PersistenceEntropy_0,
landscape1_0, landscape1_1,
landscape2_0, landscape2_1,
betti_0, betti_1
```

The first four columns are electrode-specific. The final seven are
whole-graph persistent-homology summaries repeated across all 29 nodes. This
broadcast is an explicit inferred interpretation of the manuscript's stated
`29 × 11` input, not a confirmed node-specific PH mapping.

The model is `EEGGCN1` (Zhang et al. 2026, Table 2): one `DenseGCNConv`
(11→14) over a dense adjacency, BatchNorm/LeakyReLU(0.01)/dropout, all 29
node embeddings flattened into one vector, then two fully connected layers
(406→29→2) ending in log-softmax. The adjacency is derived fresh from the
saved PLV matrices by thresholding at `>=0.3` and binarizing to 0/1 — the
manuscript's own ablation-selected graph construction — not the top-k
weighted adjacency saved as `adjacency_matrices.npy`. Training uses
`nn.NLLLoss` (paired with the log-softmax output) and the manuscript's
gamma+PLV-tuned defaults, `learning_rate=0.005` and `dropout=0.4`. There is
no dataset selector and no separate graph-feature input in this experiment.

## Run one held-out subject

```bash
PYTHONPATH=/workspaces/EEG python \
  -m src.ourexperimentversionthree.training.train \
  --test-subject 1 \
  --run-name subject01_seed42 \
  --seed 42 \
  --seed-strategy shared \
  --epochs 5
```

## Run all LOSO folds

```bash
PYTHONPATH=/workspaces/EEG python \
  -m src.ourexperimentversionthree.training.train \
  --all-subjects \
  --run-name broadcast11_seed42 \
  --seed 42 \
  --seed-strategy shared \
  --epochs 50 \
  --overwrite
```

Omitting `--run-name` creates a UTC timestamp name. A non-empty run directory
is protected unless `--overwrite` is passed explicitly.

## Outputs

```text
outputs/<run_name>/
  run_manifest.json
  loso_results.json
  loso_results.csv
  subject_01/
    best_model.pt
    result.json
```

Checkpoints include the fitted training-only normalization, model and training
configuration, dataset metadata, subject split, resolved device, and metrics.

## Methodology

- Subjects, rather than individual graphs, are separated across training,
  validation, and test partitions.
- All 11 feature columns are normalized using training subjects only.
- Validation loss selects the best epoch; the held-out test subject is
  evaluated after selection.
- `shared` gives each fold the same initialization seed. `per_fold` uses
  `base_seed + test_subject_id`.
- Human-readable progress is written to stderr and final JSON to stdout.

The dataset-generation assumption and verification commands are documented in
`src/datautils/PlvLiu2024Broadcast11/DATASET_GENERATION.md`.
