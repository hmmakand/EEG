# Graph training from saved within-subject folds

`datacustom` reads the finalized signal bundle in `datasyn/fold_first/`. The
preparation stage already chose all folds and generated a separate synthetic
set for each within-subject fold. Graph creation and classifier training never
split the signals again.

## Run supervised training

From the repository root, use all subjects and saved folds, 100 epochs per fold,
and batch size 8:

```bash
source .venv/bin/activate
python -u -m src.ourexpv17.train.train_syn \
  --bundle src/ourexpv17/datasyn/fold_first \
  --epochs 100 --batch-size 8 --device cuda \
  --output src/ourexpv17/output/PLV_SavedWithinSubject_FoldSpecificSynthetic_100epochs_Batch8.json
```

For a short CPU check, use `--subjects 1 --epochs 1 --device cpu --quiet`.
This still executes every saved fold for S1. From a standalone copy, use
`python train/train_syn.py` with the same options and suitable paths.

| Option | Default |
| --- | --- |
| `--bundle` | This experiment's `datasyn/fold_first/` |
| `--subjects` | All subjects in the saved protocol |
| `--epochs` | 100 per fold |
| `--batch-size` | 8 |
| `--device` | `auto` |
| `--output` | Timestamped JSON under this experiment's `output/` |
| `--quiet` | Hide epoch progress bars |

There are no `--folds`, split-seed, `--protocol`, `--original-dir`,
`--synthetic-dir`, or combined-validation/test options. Change preparation and
regenerate affected synthetic data to change fold membership.

## Data and graph rules

The loader requires `protocols/within_subject.csv` and a completed
**within-subject** entry in `generation_status.json`. Cross-subject generation
may still be pending. An unfinished plan is not accepted as a finalized dataset.
Cross-subject classifier loading is not implemented in this step.

For the current 144-original/100-synthetic preparation:

| Folds per subject | Original training | Synthetic training | Total training | Original testing |
| --- | ---: | ---: | ---: | ---: |
| 1–4 | 129 | 100 | 229 | 15 |
| 5–10 | 130 | 100 | 230 | 14 |

Each original is tested once. A fold only uses its own generated signals.
Diffusion's internal validation originals are part of that fold's permitted
classifier training set; no separate classifier validation loader is created.

The adapter verifies prepared artifact hashes, finalized-manifest equality with
the saved plan, original and synthetic identities, labels and shapes, and each
fold generator's permitted fitting/validation inputs. Signal and generation
manifest hashes are checked when loading each fold. Classifier loading does not
load diffusion models or their checkpoints and does not require `timm`.

Signals already have original amplitude scale. Every original and synthetic
trial uses the same `dataset.build_trial_graph()` configuration: 8–30 Hz filter,
eight bandpower features, PLV, and threshold 0.35 with legacy graph edges. Defaults
produce `x=[22, 8]`, scalar `y`, and `edge_index=[2, 484]`. No extra scaling,
trial extraction, or legacy interleaving is applied.

Original graphs are built once for the current subject and reused across folds;
synthetic graphs are built per fold. Graphs carry `sample_id`, `is_synthetic`,
and `generator_job` alongside standard node/edge/label and subject/session/trial
fields. Original graphs do not carry a permanent partition because their role
changes between folds. Membership and row order come exclusively from the manifest.
Graphs remain in memory; no graph dataset is written.

```python
from src.ourexpv17.datacustom import SavedWithinSubject

saved = SavedWithinSubject()
fold = saved.load_fold(subject=1, fold=1)
train_loader, test_loader = fold.loaders(batch_size=8)
```

## Learning and reporting

`config_syn.py` owns classifier settings (100 epochs, batch 8). Learning behavior
matches the standard supervised trainer: fresh GAT and Adam per fold, learning
rate 0.001, cross-entropy, hidden channels 22, three heads, dropout 0.5, model seed
12345, and no loader shuffling. The synthetic orchestrator calls the shared
`setup.py`, `engine.py`, selection, metric, and reporting helpers. It never calls
the standard splitting code. At batch 8 each full-data fold has 29 optimizer
updates per epoch.

Evaluate training and testing after every epoch. Select the highest test accuracy
and retain the first epoch on ties, including all-zero accuracy. Fold accuracy,
balanced accuracy, macro F1, and confusion matrix all describe that selected epoch.
This is test-based epoch selection, not an independent final test estimate.
The earlier reused-generator leakage does not apply to these fold-specific
signals when their saved fitting/validation memberships pass the checks.

The results JSON contains `subjects` with the standard fold result fields and
per-subject mean/max/min accuracy, balanced accuracy and macro F1 summaries, and
summed confusion matrices. Each fold also records counts, ordered sample IDs,
optimizer updates, and source hashes. The overall `summary` averages subject
means and sums confusion matrices. Settings, manifest hash, saved split rules,
executed subjects, device, and package versions are included.

Only the result JSON is saved; no classifier checkpoints or resume support are
added. An explicitly reused output filename is overwritten. Without `--output`,
a timestamp avoids replacing previous reports.

The standard `train/train.py` workflow remains separate and unchanged.

## Verify

```bash
python -m unittest discover -s src/ourexpv17/tests -p test_training_syn.py -v
```
