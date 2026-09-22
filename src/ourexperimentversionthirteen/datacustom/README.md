# Graph training with original and synthetic EEG

## Within-subject folds with existing synthetic data

Use `--protocol kfold` to combine each subject's saved original training,
validation, and testing partitions (144 trials), then apply 10-fold `KFold`
with shuffle enabled and seed 42 **only to original trials**. Originals are
ordered by session ID and original trial index before splitting. This is
independent of the saved partition order; it does not reproduce the legacy
trainer's interleaved trial ordering or necessarily its exact fold memberships.

Each fold trains a fresh classifier on 129 or 130 original trials plus all 100
existing synthetic trials (229 or 230 total), and tests on 15 or 14 originals.
Every original is tested exactly once across folds. There is no validation set.
Saved EEG files and their original assignments are unchanged. Graphs are built
once per subject in memory and reused across folds.

The standard `experiment.run_fold()` supplies the training loop: GAT, Adam
at 0.001, cross-entropy, model seed 12345, no loader shuffling, and highest test
accuracy across epochs, with earliest ties. Training order is original fold
order followed by synthetic manifest order. At batch size 8, each fold has 29
optimizer updates per epoch (2,900 over 100 epochs).

From the repository root, run all eight subjects for 100 epochs per fold:

```bash
source .venv/bin/activate
python -u -m src.ourexperimentversionthirteen.train.train_syn \
  --protocol kfold --folds 10 \
  --original-dir src/ourexperimentversionthirteen/datasyn/original \
  --synthetic-dir src/ourexperimentversionthirteen/datasyn/synthetic/eegdiffuser_100epochs_20260921T080224Z \
  --subjects 1 2 3 5 6 7 8 9 \
  --epochs 100 --batch-size 8 --device cuda \
  --output src/ourexperimentversionthirteen/output/PLV_Original144_Synthetic100_10fold_100epochs_Batch8.json
```

Omit `--synthetic-dir` for original-only folds with identical original fold
membership. For a short CPU check add `--subjects 1 --epochs 1 --device cpu`.
The standalone entry point is `python train/train_syn.py` with suitable paths.

JSON contains each fold's selected epoch, accuracy, balanced accuracy, macro F1,
confusion matrix, sample IDs, class counts, and optimizer update counts. Subject
summaries average fold scores; the overall summary averages subject scores.
Confusion matrices are summed. Source hashes, settings, protocol, and limitations
are saved. No model checkpoints or graph datasets are written.

**Known limitations:** the existing diffusion model used some originals that
now enter test folds. Graph conversion does not remove this indirect leakage.
The best epoch is also selected using test-fold accuracy. This is an exploratory
protocol, not a leakage-free independent test estimate. Both limitations are
recorded in the JSON. Future leakage-free folds require fold-specific diffusion
training and separate epoch selection data.

## Existing fixed-partition mode

The default remains `--protocol fixed`; the following describes that mode.

`datacustom` builds graphs in memory from the existing saved EEG segments. It
returns the same PyTorch Geometric `Data` objects as `dataset/`, grouped into the
three saved partitions. No graph dataset is written to disk and no trials are
resplit.

## Data flow

For each subject:

| Partition | Original-only | Original + synthetic |
| --- | ---: | ---: |
| Training | 100 original | 100 original + 100 synthetic |
| Validation | 22 original | The same 22 original |
| Testing | 22 original | The same 22 original |

All 100 original training trials are available to the classifier. The diffusion
model's internal 90/10 training/validation split is not used here. Synthetic data
is added only to training, from one explicitly chosen completed generation run.
The adapter checks original and synthetic artifact hashes, original-dataset
provenance, identities, labels, partition order, and array dimensions.

Every trial uses `dataset.build_trial_graph()` with one shared `DatasetConfig`:
8–30 Hz filtering, eight bandpowers, PLV, and the configured edge construction.
Signals already have original amplitude scale; no diffusion scaling is applied.
Trial extraction and `legacy_interleave` are not repeated. Defaults produce
`x=[22, 8]`, scalar `y`, and `edge_index=[2, 484]` in legacy graph mode, including
its `(0, 0)` placeholders for rejected connections.

```python
from src.ourexperimentversionthirteen.datacustom import build_subject_dataset
from src.ourexperimentversionthirteen.dataset import default_dataset_config
from src.ourexperimentversionthirteen.train.config_syn import SynTrainingConfig
from src.ourexperimentversionthirteen.train.splitting_syn import build_loaders

partitions = build_subject_dataset(
    original_dir='src/ourexperimentversionthirteen/datasyn/original',
    synthetic_dir='src/ourexperimentversionthirteen/datasyn/synthetic/eegdiffuser_100epochs_20260921T080224Z',
    subject_number=1,
    dataset_config=default_dataset_config(),
)
train_loader, validation_loader, test_loader = build_loaders(
    partitions, SynTrainingConfig(),
)
```

Omit `synthetic_dir` for the original-only condition. Original-only loading does
not need the diffusion model or its dependencies.

Each graph retains `x`, `edge_index`, `y`, `subject_id`, `session_id`, and
`trial_id`, with `is_synthetic`, `sample_id`, `partition`, and `source_run` added
consistently for batching. Synthetic graphs have `session_id=-1`, their own
trial IDs, and namespaced sample identities; no original trial identity is
invented for them.

## Run the two comparable conditions

From the repository root:

```bash
source .venv/bin/activate

# Original-only baseline with the saved fixed partitions.
python -m src.ourexperimentversionthirteen.train.train_syn \
  --original-dir src/ourexperimentversionthirteen/datasyn/original \
  --epochs 249 --device auto

# Same learning settings and held-out partitions, plus synthetic training trials.
python -m src.ourexperimentversionthirteen.train.train_syn \
  --original-dir src/ourexperimentversionthirteen/datasyn/original \
  --synthetic-dir src/ourexperimentversionthirteen/datasyn/synthetic/eegdiffuser_100epochs_20260921T080224Z \
  --epochs 249 --device auto
```

Use the same epoch count for both conditions; change both commands to
`--epochs 100` if desired. The generator's 100-epoch run and the classifier's
training duration are separate settings. Add `--subjects 1 --epochs 1 --device cpu
--quiet` for a short check. `--batch-size` and `--output` are also supported.

From a standalone version-thirteen folder, use `python train/train_syn.py` with
the same options and suitable paths. No imports from version ten or `temp` are
needed. No synthetic directory is selected automatically.

## Training rules

- `config_syn.py` takes its learning defaults from the existing `train/config.py`:
  249 epochs, batch size 32, learning rate 0.001, hidden channels 22, three heads,
  model seed 12345, and device auto. The unchanged model retains dropout 0.5.
- `setup.py` supplies the same GAT, Adam optimizer, and cross-entropy loss.
  `engine.py` supplies the unchanged `train_epoch()` and `evaluate()` functions.
- `splitting_syn.py` creates three loaders without making folds or assigning
  partitions. It rejects repeated identities, mixed subjects, and synthetic
  validation/test graphs.
- The training list is mixed once using SHA-256 ordering keyed by mix seed 42,
  subject ID, and sample ID. Real trials preserve their relative order across
  original-only and augmented conditions. Per-epoch loader shuffling defaults to
  off, matching the standard setting. Validation and test ordering is unchanged.
- Each subject starts from a freshly initialized model with the same model seed.
- Evaluate training and validation accuracy after each epoch. Select strictly
  higher validation accuracy, retaining the earliest epoch on ties, including
  when all epochs have zero accuracy.
- Hold a copy of the selected model weights in memory. After training, restore
  those weights and evaluate testing once. Test results never select the epoch.

At batch size 32, 100 training graphs give 4 optimizer updates per epoch, while
200 give 7. Results record this difference; equal epochs are not equal numbers
of updates. These two fixed-partition conditions are directly comparable to
one another. The earlier KFold results use a different evaluation protocol.

## Results and responsibilities

Only a results JSON is saved. Its default path is under version thirteen's
`output/` with condition, threshold, epoch count, and timestamp in its name.
The JSON records per-subject test accuracy, balanced accuracy, macro F1,
confusion matrices, selected validation epoch, class counts, optimizer updates,
training/validation accuracy history, graph sample identities, initial training
order, source hashes, settings, device, and library versions. The summary uses
unweighted subject averages and a summed confusion matrix. Existing explicit
`--output` paths are overwritten, as in the standard trainer.

Graphs and selected weights remain in memory; this workflow does not add disk
checkpoints or resume support.

| File | Responsibility |
| --- | --- |
| `datacustom/dataset.py` | Read saved EEG, verify it, and build partitioned graphs |
| `train/config_syn.py` | Fixed-partition learning settings |
| `train/splitting_syn.py` | Validate graph membership, mix training once, build loaders |
| `train/experiment_syn.py` | Train per subject, select by validation, evaluate testing once |
| `train/train_syn.py` | CLI and JSON output |

The existing KFold scripts and version ten remain unchanged.

## Verify

```bash
python -m unittest discover -s src/ourexperimentversionthirteen/tests -p test_training_syn.py -v
```

Tests check exact equivalence with the existing graph builder, identical held-out
graphs across conditions, provenance rejection, loader isolation and batching,
checkpoint restoration and tie handling, single final test evaluation, both CLI
conditions, and operation from a standalone copy.
