# Version twenty-three, reporting the highest test epoch

This is a copy of [version twenty-three](../ourexpv23/README.md) with one
addition: the test trials are scored after every training epoch, so that two
results can be reported side by side.

| Score | How it is obtained | Uses test labels? |
|---|---|---|
| **Final epoch** | Accuracy after the last epoch. The number of epochs is fixed in advance. This is the result of version 23. | No |
| **Highest epoch** | Accuracy of the epoch with the highest test accuracy, chosen separately for every fold. This is how some published work reports (for example Patel et al. 2025, whose code keeps the best test accuracy of each fold). | Yes, to choose the epoch |

Data processing, model and training are unchanged. With the same seed on CPU
the final-epoch score is identical to version 23, trial for trial; on GPU it
differs by a trial or two between runs, as version 23 itself does.

## Results (eight subjects, seed 12345)

| Experiment | Final epoch | Highest epoch | Difference |
|---|---|---|---|
| Leave one subject out | 80.03% | 82.20% | +2.17 |
| Within subject, 10 folds | 79.69% | 89.06% | +9.37 |

The same two experiments with the **training labels shuffled**, so that the
model cannot learn anything about left versus right:

| Control experiment | Final epoch | Highest epoch | Difference |
|---|---|---|---|
| Leave one subject out, shuffled labels | 50.00% | 56.60% | +6.60 |
| Within subject, shuffled labels | 48.18% | 65.28% | +17.10 |

### What these numbers say

- **The final-epoch score behaves as an honest estimate.** With shuffled
  labels it is at chance (50.0% and 48.2%).
- **The highest-epoch score is not at chance when there is nothing to learn.**
  It reaches 65.3% within subject and 56.6% in LOSO with shuffled labels. That
  amount is produced by the selection alone.
- **The inflation depends on the size of the test set.** A within-subject fold
  has 14 or 15 test trials, so one trial is about 7 points and the best of 60
  epochs is often several trials above the average. A LOSO fold has 144 test
  trials and 20 epochs, so the inflation is smaller.
- **The gain of the highest-epoch score on real labels is no larger than what
  selection produces on shuffled labels** (+9.4 against +17.1 within subject,
  +2.2 against +6.6 in LOSO). There is no evidence in it of a better model;
  the model is the same one.

### Per subject

Leave one subject out (144 test trials per subject, 20 epochs):

| Held-out subject | Final epoch | Highest epoch | Best epoch |
|---|---|---|---|
| S1 | 83.33% | 85.42% | 15 |
| S2 | 72.22% | 72.22% | 11 |
| S3 | 91.67% | 93.75% | 11 |
| S5 | 68.06% | 69.44% | 12 |
| S6 | 76.39% | 79.86% | 5 |
| S7 | 66.67% | 72.22% | 8 |
| S8 | 89.58% | 90.97% | 5 |
| S9 | 92.36% | 93.75% | 6 |
| **Mean** | **80.03%** | **82.20%** | |

Within subject (10 folds of 14 or 15 test trials, 60 epochs; the best epoch is
chosen per fold, and the subject's score is taken over all 144 trials):

| Subject | Final epoch | Highest epoch | Shuffled labels, final | Shuffled labels, highest |
|---|---|---|---|---|
| S1 | 81.25% | 92.36% | 47.22% | 64.58% |
| S2 | 77.08% | 84.03% | 50.69% | 65.28% |
| S3 | 89.58% | 97.92% | 43.75% | 65.97% |
| S5 | 64.58% | 77.78% | 50.69% | 68.75% |
| S6 | 64.58% | 77.78% | 49.31% | 66.67% |
| S7 | 68.75% | 82.64% | 43.06% | 63.19% |
| S8 | 95.83% | 100.00% | 52.08% | 63.19% |
| S9 | 95.83% | 100.00% | 48.61% | 64.58% |
| **Mean** | **79.69%** | **89.06%** | **48.18%** | **65.28%** |

All four runs use one seed. Version 23 measured a seed-to-seed spread of about
0.7 points for LOSO over five seeds.

### All nine subjects (subject 4 included, `--all-subjects`)

| Experiment | Final epoch | Highest epoch | Difference |
|---|---|---|---|
| Leave one subject out | 77.93% | 80.94% | +3.01 |
| Within subject, 10 folds | 79.32% | 88.19% | +8.87 |

| Subject | LOSO, final | LOSO, highest | Within, final | Within, highest |
|---|---|---|---|---|
| S1 | 81.94% | 85.42% | 83.33% | 92.36% |
| S2 | 70.14% | 71.53% | 77.08% | 82.64% |
| S3 | 87.50% | 93.75% | 88.19% | 97.92% |
| S4 | 69.44% | 73.61% | 72.92% | 81.94% |
| S5 | 67.36% | 70.14% | 65.28% | 77.78% |
| S6 | 77.08% | 80.56% | 65.28% | 79.17% |
| S7 | 68.06% | 68.75% | 69.44% | 81.94% |
| S8 | 89.58% | 92.36% | 97.22% | 100.00% |
| S9 | 90.28% | 92.36% | 95.14% | 100.00% |
| **Mean** | **77.93%** | **80.94%** | **79.32%** | **88.19%** |

One seed (12345). The shuffled-label control was not run for nine subjects.
The nine-subject means average over a different set of test subjects and are
not directly comparable with the eight-subject means. Result files:
`Version23ReportingHighest_{LOSO,Within}_AllSubjects_seed12345.json`.

## How to report this

- Give the final-epoch score as the result: **80.0% LOSO, 79.7% within subject**.
- If the highest-epoch score is shown, label it as "highest test accuracy over
  the epochs, selected per fold on the test trials", and show the
  shuffled-label control next to it (56.6% LOSO, 65.3% within subject).
- When comparing with a paper that reports the highest test epoch, compare its
  number with our highest-epoch number (82.2% LOSO, 89.1% within subject), not
  with our final-epoch number. The two kinds of score are not comparable.

## Run it

From the repository root:

```bash
source .venv/bin/activate
RAW=/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv23reportinghighest/outputs
RUN=src.ourexpv23reportinghighest.train

python -u -m $RUN.run_loso   --data-dir "$RAW" --output "$OUT/my_loso.json"
python -u -m $RUN.run_within --data-dir "$RAW" --output "$OUT/my_within.json"

# Control runs: the training labels are shuffled.
python -u -m $RUN.run_loso   --data-dir "$RAW" --output "$OUT/my_loso_shuffled.json"   --shuffle-labels
python -u -m $RUN.run_within --data-dir "$RAW" --output "$OUT/my_within_shuffled.json" --shuffle-labels
```

On a GTX 1060 a LOSO run takes about 7 minutes and a within-subject run about
25 minutes. The other options are those of version 23: `--subjects`,
`--all-subjects`, `--seed`, `--device`, `--quiet`, and for the within-subject
run `--folds`, `--split-seed`, `--epochs`.

Tests: `python -m unittest discover -s src/ourexpv23reportinghighest/tests -t .`

## The result file

```
summary.final_epoch      mean accuracy, balanced accuracy, macro F1, summed confusion matrix
summary.highest_epoch    the same, for the highest-epoch scores
subjects.<n>.final_epoch / highest_epoch
subjects.<n>.highest_epoch.epoch        LOSO: the epoch that was chosen (1 = first)
subjects.<n>.test_accuracy_per_epoch    LOSO: test accuracy after every epoch
subjects.<n>.folds[...]                 within subject: the same three entries for every fold
shuffled_training_labels                true for a control run
evaluation                              the definition of the two scores
```

Result files of the runs above are in
`databci/bciupperlimbactionexperiment/ourexpv23reportinghighest/outputs/`:
`Version23ReportingHighest_{LOSO,Within}_seed12345.json` and
`Version23ReportingHighest_{LOSO,Within}_ShuffledLabels_seed12345.json`.

## What differs from version 23 in the code

| File | Change |
|---|---|
| [train/engine.py](train/engine.py) | `test_logits` (logits after an epoch), `ensemble_predictions` (probability averaging and centring, as in version 23), `score_epochs` (the two scores) |
| [train/loso.py](train/loso.py) | `train_model` returns the test logits of every epoch; `run_fold` scores every epoch; `shuffled_labels` for the control run |
| [train/within.py](train/within.py) | Both scores are pooled over the folds of a subject |
| [train/metrics.py](train/metrics.py) | `summarize_both` |
| [train/config.py](train/config.py) | `shuffle_training_labels` |
| [train/run_loso.py](train/run_loso.py), [train/run_within.py](train/run_within.py) | `--shuffle-labels` |

`data/` and `model/` are unchanged. For the pipeline, the model, and what was
tried and did not help, see the [README of version 23](../ourexpv23/README.md).

Two details of the scoring:

- With an ensemble of three models, the epoch is chosen for the ensemble: the
  three models' probabilities are averaged at each epoch, and the best epoch
  of that average is taken.
- On a tie, the earliest epoch with the highest accuracy is taken.
