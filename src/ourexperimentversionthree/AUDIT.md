# Implementation audit: experiment version three

Audit date: 2026-09-03

## Scope and evidence

This audit covers the executable code in `data/`, `model/`, and `training/`,
the generator in `src/datautils/PlvLiu2024Broadcast11`, the saved dataset and
its metadata, the automated tests, and the recorded 50-fold run. The
experiment README was explicitly excluded from the audit.

Verification performed during the audit:

- `PYTHONPATH=/workspaces/EEG pytest -q src/ourexperimentversionthree`:
  **22 passed, 7 subtests passed** (two third-party deprecation warnings).
- `python -m src.datautils.PlvLiu2024Broadcast11.verify_dataset`:
  **passed** for the canonical saved dataset.
- Direct array checks confirmed 2,000 graphs, 50 subjects with 40 graphs each,
  20 examples of each class per subject, zero diagonal in the saved adjacency,
  and exact broadcasting of the seven topology values.

## Executive conclusion

The implementation is internally coherent and has good subject-level leakage
protection. It trains the intended fixed-size dense GCN and evaluates it using
a reproducible subject-disjoint LOSO procedure. The saved run, however,
performs at chance: mean accuracy is **0.5055** and mean AUC is **0.4968**.
It therefore supplies no evidence that this representation generalizes to
unseen subjects.

The most important scientific limitation is not a coding failure but the
feature interpretation: seven graph-level persistent-homology descriptors are
copied onto every electrode to manufacture a 29 x 11 node matrix. That is an
explicitly inferred representation, not a demonstrated node-specific mapping.
This duplication should be prominent in any methods section and should be
tested against simpler baselines.

## Dataset: structure and operation

### Source and sample unit

The canonical input is generated from 31--40 Hz preprocessed Liu2024 EEG.
Each source example is a `(29, 2000)` signal window: 29 fixed, ordered EEG
channels and 2,000 samples at 500 Hz (four seconds). There are 2,000 examples
from 50 subjects. Every subject contributes 40 trials: 20 left-hand examples
(label 0) and 20 right-hand examples (label 1).

One EEG window becomes one graph. Consequently:

- graphs: `2,000`;
- node-feature tensor: `(2000, 29, 11)`, `float32`;
- PLV tensor: `(2000, 29, 29)`, `float32`;
- saved binary adjacency: `(2000, 29, 29)`, `float32`;
- labels, subject IDs, trial IDs, and global window IDs: `(2000,)`, `int64`;
- separate model-level graph features: none.

The 29 node positions have a stable electrode order (FP1 through O2). This is
important because the model eventually flattens nodes, so it relies on that
order and is not permutation invariant.

### Connectivity and topology

For each window, the generator trims 125 samples (0.25 seconds) from each end
before Hilbert-phase PLV calculation. It retains the complete PLV matrix for
topological calculations. The graph adjacency is obtained by retaining PLV
values at least 0.3, binarizing retained connections, and removing the
diagonal. Saved graphs contain 53--387 undirected edges (mean 144.201); 402 of
2,000 graphs contain at least one isolated node.

At load time, the experiment regenerates adjacency from `plv >= 0.3` rather
than reading `adjacency_matrices.npy`. This initially sets the PLV diagonal to
one, whereas the saved adjacency has a zero diagonal. In the present model the
difference is behaviorally neutral: `DenseGCNConv` overwrites the diagonal
with self-loops. It is still needless duplication and creates two sources of
truth; loading the saved adjacency, or explicitly clearing the regenerated
diagonal and testing equivalence, would be clearer.

### Eleven feature columns

Four columns are local to each electrode:

1. mean absolute value (MAV);
2. standard deviation (population convention, `ddof=0`);
3. mean spectral density in 31--40 Hz using Welch/Hann settings;
4. normalized betweenness centrality on the thresholded binary graph.

Seven values are calculated once for the complete PLV graph and then repeated
unchanged across all 29 nodes:

1. persistence entropy for homology dimension 0;
2. first two landscape summaries for dimension 0;
3. first two landscape summaries for dimension 1;
4. Betti summaries for dimensions 0 and 1.

Thus each sample is constructed as `(29, 4 local)` plus a broadcast of
`(7 global)` to produce `(29, 11)`. The audit copy
`broadcast_source_features.npy` is not passed separately to the model.

### Split and normalization

For every test subject, that subject's 40 graphs form the test set. Five of
the remaining subjects are selected reproducibly for validation using a
per-fold RNG seeded with `base_seed + test_subject_id`; the other 44 subjects
form training. The normal fold sizes are therefore:

| Partition | Subjects | Graphs |
|---|---:|---:|
| Training | 44 | 1,760 |
| Validation | 5 | 200 |
| Test | 1 | 40 |

All 11 columns are standardized with one mean and standard deviation per
column, fitted across graphs and nodes from training subjects only. Those
statistics are reused for validation/test and saved in each checkpoint. Graph
indices and subject sets are checked for disjointness and complete coverage.
This is the strongest part of the methodology and prevents direct subject and
normalization leakage.

## Model

`EEGGCN1` accepts `x: (B, 29, 11)` and `adj: (B, 29, 29)` and has this path:

```text
DenseGCNConv 11 -> 14
  -> BatchNorm over the 14 channels across all nodes in the batch
  -> LeakyReLU(0.01)
  -> Dropout(0.4)
  -> flatten 29 x 14 = 406
  -> Linear 406 -> 29
  -> LeakyReLU(0.01)
  -> Dropout(0.4)
  -> Linear 29 -> 2
  -> log-softmax
```

The model has 12,059 trainable parameters. `DenseGCNConv` adds self-loops and
symmetrically degree-normalizes adjacency. Flattening preserves electrode
identity and lets the classifier weight every electrode position differently,
but it also means the network cannot accept another montage or node ordering
without retraining.

Because the seven global features are identical at every node, 203 of the 319
input scalar positions per graph are duplicated. Graph convolution mixes these
constants according to degree normalization, so their post-convolution values
can partly encode node degree as well as the original global descriptor. This
is a consequential modeling choice, not simply a storage detail.

## Training and evaluation strategy

Each fold resets Python, NumPy, and PyTorch seeds. With the recorded `shared`
policy, every fold uses seed 42 for model initialization; validation selection
and loader shuffling use the base seed. Training uses:

- AdamW, learning rate 0.005 and weight decay `1e-4`;
- `NLLLoss`, correctly paired with model log-probabilities;
- batch size 32;
- gradient norm clipping at 1.0;
- at most 50 epochs;
- validation-loss checkpointing with minimum improvement `1e-4`;
- early stopping after 10 unimproved epochs.

Only after selection is the best validation-loss state loaded and evaluated on
the held-out subject. Metrics are loss, accuracy, positive-class F1, recall,
precision, and ROC AUC. LOSO summaries weight each subject equally, which is
also example-equivalent here because every subject has 40 examples.

Checkpoints contain model state, model/training configuration, normalization,
dataset metadata/path, split indices, seed, history, and metrics. The run
manifest also records package versions, device, command, Git revision, and
dirty-worktree status.

## Findings and recommendations

### High: the recorded experiment is at chance

The completed 50-fold run reports accuracy `0.5055 +/- 0.0511` and AUC
`0.4968 +/- 0.0909`. Mean loss is 0.7002, close to or worse than the balanced
binary baseline of `ln(2) = 0.6931`. F1 (`0.4606 +/- 0.2201`) and recall
(`0.5540 +/- 0.3514`) are highly unstable; several folds predict no positive
examples while others predict nearly everything positive.

Recommendation: do not present this run as successful reproduction evidence.
Compare against majority, logistic/MLP, local-four-only, global-seven-only,
and adjacency-only baselines. Report uncertainty across subjects and repeat
the complete protocol across multiple initialization seeds.

### High: the broadcast representation is scientifically underdetermined

The seven topology scalars are whole-graph quantities, not electrode-specific
measurements. Repeating them across nodes meets the input width mechanically
but does not establish that this is the intended semantics. It also gives the
global block 29-fold representation before flattening.

Recommendation: explicitly label the experiment an inferred implementation.
Run ablations and preferably give global features a true graph-level branch,
then compare it with broadcasting. Do not claim manuscript fidelity beyond
the parts that are actually specified.

### Medium: validation is strong on shape but incomplete on semantics

The experiment validator checks exact shapes/dtypes, finite values, overall
class set, subjects, 40 graphs per subject, metadata, and exact topology
broadcasting. It does not verify PLV symmetry/range/diagonal, exact equality of
saved adjacency to thresholded PLV, binary adjacency/no self-edges, per-subject
20/20 class balance, trial-ID uniqueness, or correspondence of the four local
features to source signals. Generator-time validation covers some of these,
but training trusts already-saved arrays.

Recommendation: add these invariants to `data/validation.py`, including hashes
or regeneration checks when strict provenance is required.

### Medium: one seed is not enough, and CUDA is not strictly deterministic

The code seeds common RNGs but does not enable deterministic PyTorch algorithms
or record all determinism settings. The recorded run uses CUDA and a single
initialization seed. Exact bitwise reproduction is therefore not guaranteed,
and the reported variance is across subjects, not across training randomness.

Recommendation: run at least several prespecified seeds, report subject- and
seed-level uncertainty, and either enable deterministic execution or document
that results are statistically rather than bitwise reproducible.

### Medium: graph construction has two sources of truth

The loader ignores the validated saved adjacency and reconstructs it from PLV
with an internal default threshold. The threshold is not exposed in
`TrainingConfig`; its effective value is only indirectly recoverable from
dataset metadata and code. The loader comment also characterizes the saved
adjacency as top-k weighted, although this generated dataset stores thresholded
binary adjacency.

Recommendation: use the saved validated adjacency for this fixed experiment,
or make graph construction an explicit immutable experiment configuration and
assert regenerated equality. Correct the stale loader description.

### Low: package execution assumptions affect test discovery

Tests pass when the repository is on `PYTHONPATH`. Invoking `pytest` without
that environment setting fails collection because absolute `src.*` imports
cannot be resolved in the current environment.

Recommendation: install the project in editable mode in development/CI or add
a standard test configuration that puts the repository root on the import
path.

### Low: reporting can be strengthened

The summary uses population standard deviation (`ddof=0`) and gives no
confidence intervals, confusion totals, calibration measures, or pooled ROC.
The output is adequate for debugging but thin for a scientific result.

Recommendation: retain fold predictions, report confidence intervals and a
confusion matrix, state the positive class and averaging convention, and
distinguish subject variation from seed variation.

## Overall assessment

Code quality and leakage discipline are good, and the implementation is
test-covered and operational. The current evidence nevertheless says that the
chosen representation/model does not generalize beyond chance. The priority
should be semantic validation of the seven broadcast features and controlled
ablations, followed by multi-seed evaluation—not further tuning against the
held-out LOSO subjects.
