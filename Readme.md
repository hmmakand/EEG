Ourversionten -- base line (corrected: GATv2 on PLV 0.35 graphs, fair evaluation)
ourversionthirteen -- baseline with eegdiffuser 
ourversionfourteen -- baseline within cross subject -- pc data generation 
ourexpv15 -- baseline within cross subject with global inclusion
ourexpv16 -- baseline within cross subject with global inclusion+self-conditioning - autodl batch 16 for data generation
ourexpv17 -- baseline within cross subject without global inclusion+self-conditioning - autodl batch 16 for data generation
ourexpv18 -- is copied from ourecpv16 -- and include the custom model.
ourexpv23 -- best model only (temporal encoder + graph head + wide path), reduced code base
**New improvement over 80 percent reach first time in version 20**

## Current results: version 10 (baseline) and version 23 (best model)

Left hand versus right hand, BCI Competition IV 2a, subjects 1, 2, 3, 5, 6, 7, 8, 9.
Both versions use the same trials, signal processing, splits, seed (12345) and
evaluation; they differ in the model and in what the graph carries. Every score
is taken once at the final epoch, with nothing selected on test trials.

### Leave one subject out (LOSO)

Train on seven subjects, test on the eighth. 20 epochs, three models per fold.

| Subject | Version 10 | Version 23 |
|---|---:|---:|
| S1 | 54.17% | 83.33% |
| S2 | 52.08% | 71.53% |
| S3 | 51.39% | 91.67% |
| S5 | 49.31% | 68.75% |
| S6 | 53.47% | 77.78% |
| S7 | 56.94% | 66.67% |
| S8 | 51.39% | 89.58% |
| S9 | 59.72% | 92.36% |
| **Mean accuracy** | **53.56%** | **80.21%** |
| Balanced accuracy | 53.56% | 80.21% |
| Macro F1 | 53.22% | 80.20% |

Result files: `ourexpv10/outputs/my_loso.json` and `ourexpv23/outputs/my_run.json`
in `databci/bciupperlimbactionexperiment/`.

### Within subject

Stratified 10-fold inside each subject, 60 epochs, three models per fold.

| Subject | Version 10 | Version 23 |
|---|---:|---:|
| S1 | 56.94% | 82.64% |
| S2 | 53.47% | 77.08% |
| S3 | 60.42% | 88.19% |
| S5 | 50.69% | 65.97% |
| S6 | 51.39% | 63.89% |
| S7 | 56.94% | 69.44% |
| S8 | 53.47% | 95.83% |
| S9 | 63.89% | 95.83% |
| **Mean accuracy** | **55.90%** | **79.86%** |
| Balanced accuracy | 55.90% | 79.86% |
| Macro F1 | 55.74% | 79.80% |

Result files: `ourexpv10/outputs/my_within.json` and `ourexpv23/outputs/my_within.json`.

### All nine subjects (subject 4 included)

Run with `--all-subjects`, seed 12345. In LOSO each fold trains on eight
subjects (1152 trials) and tests on the ninth.

| Subject | Version 10 LOSO | Version 23 LOSO | Version 10 within | Version 23 within |
|---|---:|---:|---:|---:|
| S1 | 55.56% | 81.94% | 68.75% | 83.33% |
| S2 | 58.33% | 70.83% | 53.47% | 77.08% |
| S3 | 52.78% | 87.50% | 60.42% | 89.58% |
| S4 | 46.53% | 69.44% | 45.14% | 72.92% |
| S5 | 47.22% | 68.75% | 50.69% | 65.28% |
| S6 | 50.00% | 76.39% | 54.86% | 65.28% |
| S7 | 58.33% | 68.06% | 53.47% | 69.44% |
| S8 | 59.03% | 89.58% | 53.47% | 96.53% |
| S9 | 61.11% | 90.28% | 59.03% | 95.83% |
| **Mean accuracy** | **54.32%** | **78.09%** | **55.48%** | **79.48%** |
| Balanced accuracy | 54.32% | 78.09% | 55.48% | 79.48% |
| Macro F1 | 53.98% | 78.05% | 55.31% | 79.41% |

Result files: `Version10Fixed_LOSO_AllSubjects_seed12345.json`,
`Version10Fixed_Within_AllSubjects_seed12345.json` in `ourexpv10/outputs/`, and
`Version23_LOSO_AllSubjects_seed12345.json`,
`Version23_Within_AllSubjects_seed12345.json` in `ourexpv23/outputs/`.

The nine-subject means are not directly comparable with the eight-subject
means above: they average over one more test subject, and in LOSO the other
eight subjects are trained with one more subject.

### Notes

- These are single runs. Over five seeds the LOSO mean is 53.5% for version 10
  and 79.0% for version 23; a single run varies by about 1 to 1.5 points.
- Chance is 50%.
- Alignment and, in LOSO, logit centring use the test subject's trials without
  their labels.
- Details: `src/ourexperimentversionten/README.md` and `src/ourexpv23/README.md`.

## Earlier comparison (superseded)

The table below reported the best test epoch of each 14-trial fold. That
selection gives about 73% even with randomly shuffled labels, so these numbers
do not measure the models and must not be compared with the results above.

| Subject | Version 10 | Version 14 | Version 15 | Version 16 |
|---|---:|---:|---:|---:|
| S1 | 75.14% | 77.86% | 73.57% | 74.29% |
| S2 | 73.62% | 76.38% | 73.52% | 74.76% |
| S3 | 72.29% | 75.62% | 75.10% | 75.76% |
| S5 | 72.90% | 75.05% | 70.81% | 71.52% |
| S6 | 72.19% | 75.00% | 72.33% | 72.86% |
| S7 | 74.43% | 72.86% | 75.48% | 79.10% |
| S8 | 75.05% | 75.24% | 76.38% | 79.19% |
| S9 | 79.90% | 79.86% | 82.10% | 76.38% |
| Mean | 74.44% | 75.98% | 74.91% | 75.48% |
