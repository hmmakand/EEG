# LOSO training audit: Versions 10, 14, 16, and 17

Audit date: 2026-10-03. Scope: current standalone LOSO code, the four saved 100-epoch/batch-8 results, and the corresponding data under `/workspaces/datadisk/databci/bciupperlimbactionexperiment`.

## Conclusion

**The held-out subject is correctly excluded from classifier training in all four runs. Testing contains only that subject's original trials.** For Versions14/16/17, synthetic samples used for a fold come only from generators fitted and selected using the other seven subjects' original data.

**The core classifier training rule is the same, but the complete training conditions are not identical.** All four use the same classifier architecture, optimizer, loss, seed, fixed sample order, and final-epoch evaluation. Version10 has no synthetic data and uses a different original-trial order. Adding synthetic data increases both training-set size and optimizer updates. Versions14/16/17 have identical assignment/order rules but different synthetic signals and generator settings.

The original-plus-synthetic arrangement is valid for subject isolation. It is a fixed sequence of subject blocks, not a randomized mixture of original and synthetic trials.

## 1. Common classifier rule

| Property | All four audited runs |
|---|---|
| Cohort | S01, S02, S03, S05, S06, S07, S08, S09; S04 is excluded |
| Folds | Eight; hold out one whole subject per fold |
| Classifier initialization | Fresh model, optimizer, and loss object for every held-out subject |
| Seed reset | `torch.manual_seed(12345)` before constructing each model |
| Classifier | Three GATv2 layers, GraphNorm, global mean pooling, dropout 0.50, two-class linear output |
| Dimensions | 22 EEG nodes, eight node features, 22 hidden channels, three attention heads |
| Optimizer | Adam, learning rate 0.001; default Adam settings, no supplied weight decay |
| Loss | Unweighted cross entropy for labels 0 and 1 |
| Training | 100 epochs, batch size 8; one optimizer step per batch |
| Shuffle | Training disabled; testing disabled |
| Classifier validation | No separate validation set |
| Epoch selection | Final epoch, 100; no best-test-accuracy selection |
| Testing | One evaluation after all training epochs, `model.eval()` and `torch.no_grad()` |
| Reporting | Per-subject accuracy, balanced accuracy, macro F1, confusion matrix; mean subject metrics and summed confusion matrix |
| Classifier artifacts | Metrics and sample identities; model weights and per-epoch history are not saved |

Each epoch calls `model.train()`, loops through the training loader, calculates cross entropy, clears gradients, backpropagates, and calls `optimizer.step()`. There is no classifier learning-rate scheduler, early stopping, sample weighting, source-specific loss, or online synthetic generation. Original and synthetic graphs go through the same classifier and loss.

The model, setup helpers, epoch engine, preprocessing, feature construction, and graph-construction source files are byte-identical across the four versions. Versions14/16/17 also have identical LOSO experiment-runner files. Version10 uses a separate original-only loader and runner, with the same learning behavior.

The shared legacy `TrainingConfig` still permits `best_test_accuracy` internally. That setting is not used by the LOSO epoch loops: these loops train all epochs and evaluate once afterward. LOSO result files correctly record `selection_policy: final_epoch`. This distinction matters when reading configuration defaults rather than the actual LOSO runner.

## 2. Fold composition and training exposure

Counts below hold for every audited fold.

| Quantity | Version10 | Versions14/16/17 |
|---|---:|---:|
| Original training subjects | 7 | 7 |
| Original training trials | 1,008 | 1,008 |
| Synthetic training trials | 0 | 700 |
| Total training trials | 1,008 | 1,708 |
| Training class counts | 504 / 504 | 854 / 854 |
| Original test trials | 144 | 144 |
| Synthetic test trials | 0 | 0 |
| Test class counts | 72 / 72 | 72 / 72 |
| Training batches per epoch | 126 | 214 |
| Last training batch | 8 trials | 4 trials |
| Optimizer updates over 100 epochs | 12,600 | 21,400 |

Each training original is seen once per epoch, so 100 times during its fold. Each included synthetic trial is also seen 100 times. Synthetic signals are generated beforehand and remain fixed throughout classifier training. The seven training subjects contribute equally: 144 originals each, plus 100 synthetic trials each when augmentation is enabled.

The synthetic fraction is 700/1,708 = **40.98%**. Augmented runs make **69.84% more optimizer updates** than Version10 at the same epoch count. Consequently, equal epochs and batch size do not imply equal optimization exposure.

## 3. How samples are arranged before training

### Version10: originals only

The loader builds each subject's original graphs using the existing extraction code, then concatenates the seven training subjects in cohort order. Within each subject, `legacy_interleave` alternates class 0 and class 1:

```text
S02 originals: 0,1,0,1,... (144)
S03 originals: 0,1,0,1,... (144)
...
S09 originals: 0,1,0,1,... (144)
```

For the S01-held-out fold, training subjects are S02, S03, S05, S06, S07, S08, S09. With batch size 8, every training batch has four trials of each class. Each subject contributes 18 full batches, so these subject boundaries align with batch boundaries.

### Versions14/16/17: originals plus synthetic

`datasetsynthetic/protocols.py` creates the saved assignments first. The LOSO loader preserves the row order of `protocols/cross_subject.csv` when constructing its training list. For each included training subject, the order is:

```text
144 originals in session/trial-index order
50 synthetic class-0 trials
50 synthetic class-1 trials
```

For S01 held out, the complete training sequence is:

```text
S02 original[144] -> S02 synthetic-0[50] -> S02 synthetic-1[50]
S03 original[144] -> S03 synthetic-0[50] -> S03 synthetic-1[50]
S05 original[144] -> S05 synthetic-0[50] -> S05 synthetic-1[50]
S06 original[144] -> S06 synthetic-0[50] -> S06 synthetic-1[50]
S07 original[144] -> S07 synthetic-0[50] -> S07 synthetic-1[50]
S08 original[144] -> S08 synthetic-0[50] -> S08 synthetic-1[50]
S09 original[144] -> S09 synthetic-0[50] -> S09 synthetic-1[50]
```

S01's original trials form the test list; its synthetic trials do not enter this fold.

**The `dataset_config.trials.order: legacy_interleave` recorded in augmented outputs does not reorder the prepared manifest.** The saved originals are in session/trial-index order, and the LOSO loader retains it. Thus Version10 and the augmented versions use the same original identities and signals, but a different original-trial sequence.

The three augmented versions have identical ordered training/testing sample IDs in corresponding folds. Their synthetic IDs identify equivalent generation positions, but their synthetic signal hashes differ: identical IDs do not mean identical generated EEG.

## 4. What happens to this order during training

PyG DataLoader uses `shuffle=False`. It takes consecutive slices of eight graphs from the training list and repeats the same sequence every epoch. It does not balance classes, balance sources, interleave subjects, or rebuild a random original/synthetic mixture.

For the augmented S01-held-out fold:

- There are 214 batches per epoch, including the final four-trial batch.
- 85 batches contain synthetic trials only.
- 78 batches contain only one class, counting both original and synthetic batches.
- A subject block has 244 trials, which is not divisible by eight. Some batches cross original/synthetic boundaries or subject boundaries.
- For the first training subject, the 144 originals occupy 18 batches. The next 50 class-0 synthetic trials form six full class-0 batches and part of a mixed-class boundary batch. Class-1 synthetic trials follow.

These counts were calculated from the actual saved manifest and labels. This arrangement does not violate LOSO isolation. It does create repeated subject/source/class blocks, which may affect optimization. The audit does not establish that ordering caused the near-chance results.

## 5. Held-out subject and generator isolation

For every fold, the audit verified:

1. No training sample has the held-out subject's identity.
2. Every test sample is original, belongs to the held-out subject, and appears exactly once in that fold.
3. Training and testing sample identities are disjoint and contain no duplicates.
4. Across the eight folds, all 1,152 original trials are tested exactly once.
5. Each included synthetic sample identifies a generator belonging to a training subject.
6. Every included generator's permitted original IDs are a subset of the classifier's training original IDs.
7. Actual generator input partitions and internal fitting/validation partitions contain only that generator's subject.
8. Prepared files, generated signal files, generator manifests, internal partitions, and selected generator checkpoints match their recorded hashes.
9. Saved classifier sample order and source hashes match the corresponding bundle manifests and current artifacts.

There is one cross-subject generator job per subject, for example `cross_subject/S02`. It uses that subject's 144 original trials: **134 for generator fitting and 10 for internal validation, five validation trials per class**. It generates 100 trials, 50 per class. Generator validation is separate from classifier testing. All 144 originals remain available to the classifier whenever that subject is a training subject.

A generator for the held-out subject can exist in the bundle because that subject is a training subject in other LOSO folds. Its existence is not leakage: it and its outputs are excluded from the fold where that subject is held out. Included generators are subject-specific, not pooled generators trained on all eight subjects.

Graph caching can also contain held-out originals or previously loaded synthetic graphs. The classifier loader selects only the current fold's assignment rows. Cache membership does not determine training membership. Graph construction filters and extracts features independently per trial, and GraphNorm operates per graph; no learned dataset-wide scaler is fitted on held-out data.

## 6. Data and generator differences

All versions use the same raw MATLAB dataset fingerprints. The audit additionally compared all **1,152 original signal arrays and labels** in each prepared bundle against Version10's raw MATLAB extraction; they match exactly. Test identity sets match across all versions, although Version10's test order differs.

All runs use 250 Hz sampling, four-second trials, 22 channels, labels 1/2 mapped to 0/1, an 8–30 Hz preprocessing band, eight spectral features, PLV connectivity, and the same legacy graph construction with threshold 0.35. Version10's general dataset default is currently `rplv`, but its LOSO entry point explicitly selects `plv`, confirmed in the saved output.

Generator settings are not identical:

| Recorded generator setting | Version14 | Version16 | Version17 |
|---|---:|---:|---:|
| Generator epochs | 100 | 100 | 100 |
| Generator batch size | 16 | 1 | 1 |
| Generator seed / generation seed | 8888 / 8888 | 8888 / 8888 | 8888 / 8888 |
| Internal split seed | 42 | 42 | 42 |
| Synthetic trials per class | 50 | 50 | 50 |
| Validation trials per class | 5 | 5 | 5 |
| Explicit architecture field | Not present | `eegdiffuser_global_fusion_v1` | Not present |
| Explicit self-conditioning field | Not present | `false` | Not present |

Absent configuration fields are not evidence of a particular architecture or self-conditioning setting. The generated signals differ between all three augmented versions. Version16 uses its generation validator; Versions14/17 validate saved provenance directly without requiring the generating architecture's configuration class. The classifier partition and learning rules remain equivalent.

## 7. Saved results covered by the audit

Files are under `/workspaces/datadisk/databci/bciupperlimbactionexperiment/`:

| Version | Result file | Accuracy | Macro F1 |
|---|---|---:|---:|
| 10 | `ourexpv10/outputs/Version10_PLV_LOSO_OriginalOnly_100epochs_Batch8_20261002T121313187708502Z.json` | 51.13% | 48.22% |
| 14 | `ourexpv14/outputs/Version14_PLV_LOSO_Synthetic_100epochs_Batch8_20261002T113252560600575Z.json` | 50.69% | 44.90% |
| 16 | `ourexpv16/outputs/Version16_PLV_LOSO_Synthetic_100epochs_Batch8_20261002T101520880744451Z.json` | 50.95% | 45.91% |
| 17 | `ourexpv17/outputs/Version17_PLV_LOSO_Synthetic_100epochs_Batch8_20261002T105638487420849Z.json` | 50.61% | 46.22% |

Balanced accuracy equals accuracy because every held-out subject has 72 trials per class. These near-chance scores do not indicate successful cross-subject generalization. They also do not establish a benefit from augmentation. Order, generator outputs, and optimizer-update counts differ, so the comparison cannot isolate augmentation's effect.

## 8. Verification evidence and limits

Seven targeted LOSO regression tests passed during this audit:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest \
  src.ourexperimentversionten.tests.test_loso \
  src.ourexperimentversionforteen.tests.test_training_syn_loso \
  src.ourexpv16.tests.test_training_syn_loso \
  src.ourexpv17.tests.test_training_syn_loso
```

The audit checked all 32 recorded classifier folds, all 24 cross-subject generator jobs, prepared-artifact hashes, classifier-recorded source hashes, signal shapes/finite values, array labels/subject identities, exact original signal agreement with raw extraction, assignment order, and source-code equivalence of shared classifier components. No training settings, source data, or result files were changed.

This verifies current code and the consistency of recorded artifacts. Classifier weights, epoch histories, and historical classifier source hashes were not saved, so the audit cannot replay or independently prove every operation of the past training executions. Resetting the PyTorch seed is not a guarantee of deterministic GPU computation. A single seed is insufficient to establish statistical superiority between versions.

For a controlled follow-up, use the same original order across versions, explicitly choose whether to shuffle every epoch, and decide whether the comparison holds epochs or optimizer updates constant. Any such changes should be reported as new experiments; the current recorded runs should retain their original settings.

## 9. Code references

- Version10: [LOSO loader](ourexperimentversionten/train/loso.py), [runner](ourexperimentversionten/train/experiment_loso.py), [entry point](ourexperimentversionten/train/train_loso.py), [trial ordering](ourexperimentversionten/dataset/trials.py).
- Version14: [LOSO loader](ourexperimentversionforteen/datacustom/loso.py), [runner](ourexperimentversionforteen/train/experiment_syn_loso.py).
- Version16: [LOSO loader](ourexpv16/datacustom/loso.py), [runner](ourexpv16/train/experiment_syn_loso.py), [protocol preparation and validation](ourexpv16/datasetsynthetic/protocols.py).
- Version17: [LOSO loader](ourexpv17/datacustom/loso.py), [runner](ourexpv17/train/experiment_syn_loso.py).
- Shared behavior: [batch loader](ourexpv16/datacustom/dataset.py), [training/evaluation engine](ourexpv16/train/engine.py), [model/optimizer construction](ourexpv16/train/setup.py), [GAT model](ourexpv16/model/gat.py), [per-trial preprocessing](ourexpv16/dataset/preprocessing.py).
