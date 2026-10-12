# Version ten, reporting the highest test epoch

This is a copy of the corrected [version ten](../ourexperimentversionten/README.md)
with one addition: the test trials are scored after every training epoch, so
that two results can be reported side by side.

| Score | How it is obtained | Uses test labels? |
|---|---|---|
| **Final epoch** | Accuracy after the last epoch. The number of epochs is fixed in advance. This is the result of version 10. | No |
| **Highest epoch** | Accuracy of the epoch with the highest test accuracy, chosen separately for every fold. | Yes, to choose the epoch |

Data processing, the GAT model and training are unchanged. The scoring code is
the same as in [version 23, reporting the highest test epoch](../ourexpv23reportinghighest/README.md).

## Results (seed 12345)

| Subjects | Experiment | Final epoch | Highest epoch |
|---|---|---:|---:|
| Eight | Leave one subject out | 52.26% | 57.55% |
| Eight | Within subject, 10 folds | 56.60% | 72.74% |
| Eight | Leave one subject out, **training labels shuffled** | 52.34% | 57.47% |
| Eight | Within subject, **training labels shuffled** | 50.69% | 68.66% |
| Nine | Leave one subject out | 51.93% | 58.02% |
| Nine | Within subject, 10 folds | 56.02% | 72.45% |

- In leave-one-subject-out, real and shuffled labels give the same scores:
  this run of version 10 is not distinguishable from chance.
- Within subject, the highest-epoch score of 72.7% is 4 points above what the
  same selection gives with shuffled labels (68.7%).

Per-subject tables and the comparison with version 23 are in
[Result_comaprison.md](../../Result_comaprison.md).

## Run it

From the repository root:

```bash
source .venv/bin/activate
RAW=/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv10reportinghighest/outputs
RUN=src.ourexpv10reportinghighest.train

python -u -m $RUN.run_loso   --data-dir "$RAW" --output "$OUT/my_loso.json"
python -u -m $RUN.run_within --data-dir "$RAW" --output "$OUT/my_within.json"
```

Add `--shuffle-labels` for a control run and `--all-subjects` to include
subject 4. The other options are those of version 10. A leave-one-subject-out
run takes a few minutes and a within-subject run about 15 minutes on a
GTX 1060.

Tests: `python -m unittest discover -s src/ourexpv10reportinghighest/tests -t .`

Result files: `Version10ReportingHighest_{LOSO,Within}_seed12345.json`, and
the same names with `_ShuffledLabels` or `_AllSubjects`, in the output folder
above. Their layout is described in the
[README of the version 23 copy](../ourexpv23reportinghighest/README.md#the-result-file).
