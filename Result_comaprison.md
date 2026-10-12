# Result comparison: version 10 and version 23, all nine subjects

Task: left-hand versus right-hand motor imagery, BCI Competition IV 2a,
subjects 1 to 9, 144 trials per subject (72 per class). Chance is 50%.
All runs use seed 12345.

| Experiment | Scoring | Version 10 | Version 23 | Difference |
|---|---|---:|---:|---:|
| Leave one subject out | Last epoch | 54.32% | **78.09%** | +23.77 |
| Leave one subject out | Highest epoch | 58.02% | **80.94%** | +22.92 |
| Within subject | Last epoch | 55.48% | **79.48%** | +24.00 |
| Within subject | Highest epoch | 72.45% | **88.19%** | +15.74 |

Values are mean accuracy over the nine subjects, each subject weighted
equally. Differences are in percentage points.

## How the numbers are obtained

- **Version 10:** three GATv2 layers on a graph of electrode pairs with PLV
  above 0.35, band powers as node features
  ([src/ourexperimentversionten](src/ourexperimentversionten/README.md)).
- **Version 23:** temporal encoder on the raw signal, plus a GatedGCN/GRIT
  graph head and a linear path ([src/ourexpv23](src/ourexpv23/README.md)).
- **Leave one subject out (LOSO):** train on eight subjects, test on the
  ninth. 144 test trials per fold, 20 epochs.
- **Within subject:** stratified 10-fold cross-validation inside each subject.
  14 or 15 test trials per fold, 60 epochs. A subject's accuracy is taken over
  all its 144 trials.
- **Last epoch:** accuracy after the final training epoch. The number of
  epochs is fixed in advance; no test label is used.
- **Highest epoch:** accuracy of the epoch with the highest test accuracy,
  chosen separately for every fold. The test labels are used to choose the
  epoch, so this score is optimistically biased.

## 1. Leave one subject out

| Held-out subject | V10 last epoch | V10 highest epoch | V23 last epoch | V23 highest epoch |
|---|---:|---:|---:|---:|
| 1 | 55.56% | 58.33% | 81.94% | 85.42% |
| 2 | 58.33% | 56.94% | 70.83% | 71.53% |
| 3 | 52.78% | 60.42% | 87.50% | 93.75% |
| 4 | 46.53% | 54.17% | 69.44% | 73.61% |
| 5 | 47.22% | 59.72% | 68.75% | 70.14% |
| 6 | 50.00% | 55.56% | 76.39% | 80.56% |
| 7 | 58.33% | 56.94% | 68.06% | 68.75% |
| 8 | 59.03% | 58.33% | 89.58% | 92.36% |
| 9 | 61.11% | 61.81% | 90.28% | 92.36% |
| **Mean accuracy** | **54.32%** | **58.02%** | **78.09%** | **80.94%** |
| Mean balanced accuracy | 54.32% | 58.02% | 78.09% | 80.94% |
| Mean macro F1 | 53.98% | 57.71% | 78.05% | 80.91% |

## 2. Within subject

| Subject | V10 last epoch | V10 highest epoch | V23 last epoch | V23 highest epoch |
|---|---:|---:|---:|---:|
| 1 | 68.75% | 75.69% | 83.33% | 92.36% |
| 2 | 53.47% | 70.83% | 77.08% | 82.64% |
| 3 | 60.42% | 74.31% | 89.58% | 97.92% |
| 4 | 45.14% | 67.36% | 72.92% | 81.94% |
| 5 | 50.69% | 70.83% | 65.28% | 77.78% |
| 6 | 54.86% | 70.14% | 65.28% | 79.17% |
| 7 | 53.47% | 73.61% | 69.44% | 81.94% |
| 8 | 53.47% | 70.14% | 96.53% | 100.00% |
| 9 | 59.03% | 79.17% | 95.83% | 100.00% |
| **Mean accuracy** | **55.48%** | **72.45%** | **79.48%** | **88.19%** |
| Mean balanced accuracy | 55.48% | 72.45% | 79.48% | 88.19% |
| Mean macro F1 | 55.31% | 72.23% | 79.41% | 88.18% |

Accuracy and balanced accuracy are equal because the two classes have the
same number of trials.

## Notes for reading the tables

- **The last-epoch and highest-epoch columns come from separate runs.** The
  last-epoch columns are the runs of versions 10 and 23. The highest-epoch
  columns are runs of their reporting copies, which train in the same way and
  also score every epoch. Runs on the GPU differ by a few trials even with the
  same seed, which is why a highest-epoch value can be below the last-epoch
  value for a subject (version 10, subjects 2, 7 and 8 in LOSO). Within their
  own runs, the reporting copies' last-epoch means are 51.93% and 56.02%
  (version 10) and 77.93% and 79.32% (version 23), LOSO and within subject.
- **The highest-epoch score is not an estimate of accuracy on new data.** With
  the training labels shuffled, so that nothing can be learned, the same
  selection gave 57.5% (version 10) and 56.6% (version 23) in LOSO, and 68.7%
  and 65.3% within subject. That control was run with eight subjects (subject
  4 left out).
- **Version 10's 72.45% within subject is close to that control (68.7%).**
  Version 23's 88.19% is far above it (65.3%).
- **One seed.** Every figure here is a single run with seed 12345.

## Source files

All under `databci/bciupperlimbactionexperiment/` on the data disk.

| Result | File |
|---|---|
| V10 last epoch | `ourexpv10/outputs/Version10Fixed_{LOSO,Within}_AllSubjects_seed12345.json` |
| V10 highest epoch | `ourexpv10reportinghighest/outputs/Version10ReportingHighest_{LOSO,Within}_AllSubjects_seed12345.json` |
| V23 last epoch | `ourexpv23/outputs/Version23_{LOSO,Within}_AllSubjects_seed12345.json` |
| V23 highest epoch | `ourexpv23reportinghighest/outputs/Version23ReportingHighest_{LOSO,Within}_AllSubjects_seed12345.json` |
| Shuffled-label controls (eight subjects) | `ourexpv{10,23}reportinghighest/outputs/Version{10,23}ReportingHighest_{LOSO,Within}_ShuffledLabels_seed12345.json` |

Code: [version 10](src/ourexperimentversionten/README.md),
[version 10 reporting copy](src/ourexpv10reportinghighest/README.md),
[version 23](src/ourexpv23/README.md),
[version 23 reporting copy](src/ourexpv23reportinghighest/README.md).
