# Fold-first EEG signal preparation

The active v16 workflow extracts every original EEG trial, saves within-subject
and cross-subject fold plans, and generates synthetic signals using only each
job's permitted originals. It performs no graph construction. The old
`legacy_original.py`/`legacy_config.py` fixed-split helpers are not used here.

## Prepare and generate

### Run an already prepared bundle

From the repository root, with the project Python environment active:

```bash
python -u -m src.ourexpv16.datasetsynthetic.generate \
  --bundle src/ourexpv16/datasyn/fold_first \
  --protocol within_subject \
  --epochs 100 \
  --device cuda
```

This selects all within-subject jobs (eight subjects, ten folds each with the
default preparation settings), trains each unfinished fresh model for 100 epochs,
and generates its synthetic EEG. Training, validation, and neural-network
sampling use the GPU; data loading and file writing use the CPU.

### Enable self-conditioning with global fusion

The default model uses class-label/timestep conditioning and global-token fusion.
Add `--self-conditioning` to also feed predicted clean EEG into the model during
training and sampling. The flag is saved in each job's configuration and restored
when its checkpoint is loaded. Without the flag, self-conditioning stays disabled.

For this changed model, prepare a fresh bundle and run:

```bash
python -m src.ourexpv16.datasetsynthetic.prepare \
  --data-dir data/kaggle/bci-competition-iv-data-sets-2a \
  --output src/ourexpv16/datasyn/fold_first_self_conditioning

python -u -m src.ourexpv16.datasetsynthetic.generate \
  --bundle src/ourexpv16/datasyn/fold_first_self_conditioning \
  --protocol within_subject \
  --epochs 100 \
  --device cuda \
  --self-conditioning
```

If `fold_first` has only prepared signals/folds and no generator runs, you can
use that existing bundle instead. Do not resume old global-fusion-only checkpoints
with self-conditioning enabled. Existing source-hash checks also reject runs
created before this model-source change; preserve those runs and use a fresh bundle.
When resuming the new experiment, keep `--self-conditioning` and add `--resume`.

The clean-estimate embedder starts at zero. Approximately half of training batches
use an extra detached prediction; the other batches use zero feedback. Both DDPM
and DDIM sampling reuse the preceding step's clean estimate, with matching feedback
in both classifier-free guidance branches. Validation uses the same stochastic
self-conditioning loss path; a lower loss alone does not prove better augmentation.

### Batch size

The command reads batch sizes from [eegdiffuser/config.py](eegdiffuser/config.py).
The current defaults are `batch_size = 1` for training/validation and
`sample_batch_size = 1` for synthetic generation. This CLI has no `--batch-size`
option. To use training batch size 32 for a fresh experiment, change the existing
field in `Config` to:

```python
batch_size: int = 32
```

Then run the same command above. This does not change `sample_batch_size`.
With 119–120 training trials per fold, batch size 32 gives four optimizer updates
per epoch. Larger batches change optimization; increasing epochs is not
automatically required. GPU memory capacity at batch size 32 has not been verified.
Keep saved settings unchanged when resuming an existing job; use a new bundle
for a separate experiment with different settings.

### Prepare a fresh bundle

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv16/requirements.txt
python -m src.ourexpv16.datasetsynthetic.prepare \
  --data-dir data/kaggle/bci-competition-iv-data-sets-2a \
  --output src/ourexpv16/datasyn/fold_first

python -m pip install -r src/ourexpv16/datasetsynthetic/eegdiffuser/requirements.txt
python -u -m src.ourexpv16.datasetsynthetic.generate \
  --bundle src/ourexpv16/datasyn/fold_first \
  --protocol within_subject --epochs 100 --device cuda
```

Preparation verifies and reuses an identical existing bundle. Generation verifies
and skips completed jobs with matching settings, explicit model architecture,
and model-source hash. Copied v14 generation jobs are rejected; prepare a new
bundle for the new model instead of overwriting the historical files. To resume unfinished jobs, add
`--resume` and keep the original epoch count and settings. For a fresh experiment,
choose a new `--output` directory during preparation and use that same directory
as `--bundle` during generation and graph training. Copied artifacts retain their
historical provenance; they are not automatically treated as new v16 results.

For cross-subject generation use `--protocol cross_subject`; use `--protocol both`
to process both protocols. The reference diffusion default is 1,000 epochs when
`--epochs` is omitted. Subjects/fold rules and generation counts are saved during
preparation, not chosen again during classifier training.

From a standalone copy:

```bash
python datasetsynthetic/prepare.py --data-dir /path/to/raw/matlab/files
python datasetsynthetic/generate.py --protocol within_subject --epochs 100 --device cuda
```

Paths default to `datasyn/fold_first/` inside this copy regardless of working
directory. Explicit relative paths are relative to the working directory.
Original MATLAB files are supplied separately.

## Protocol rules

Default subjects: 1, 2, 3, 5, 6, 7, 8, 9. Original signals have 1,000 samples and
22 channels (4 seconds at 250 Hz). All 144 eligible trials per subject are saved
without a permanent 70/15/15 partition. Preparation uses stable trial identities
and records exclusions, signal parameters, source hashes, and artifact hashes.

- **Within-subject:** ten shuffled KFold splits per subject, seed 42, using original
  trial order by session and event index. Each fold's diffusion model uses only
  its 129–130 permitted original training trials. Five trials per class select
  the diffusion checkpoint; 119–120 update diffusion weights. Generate 50 trials
  per class for that fold. Classifier training has 229–230 signals and testing
  has 14–15 originals.
- **Cross-subject:** leave one subject out. Generate 100 signals per subject using
  only that subject's 144 originals, internally split into 134 training and ten
  validation trials. Exclude that subject's originals and synthetic signals from
  training when it is held out. Each classifier fold has 1,708 training signals
  from seven subjects and 144 original test signals from the held-out subject.

Generation needs 80 within-subject jobs and eight reusable subject-specific
cross-subject jobs. There is no separate classifier validation partition in the
saved protocol. Diffusion validation and classifier epoch selection are separate.

## Bundle contents

```text
datasyn/fold_first/
├── original/                  # Original signals and trial identities
├── inputs/                    # Permitted originals for each diffusion job
├── synthetic/                 # Job-specific models and generated signals
├── protocols/
│   ├── within_subject.plan.csv
│   ├── cross_subject.plan.csv
│   └── within_subject.csv     # Published only after its jobs complete
├── jobs.json
├── metadata.json
└── generation_status.json
```

The cross-subject final CSV is published when its generation completes. A complete
within-subject protocol is usable even while cross-subject generation is pending.
Each final manifest references signal files relatively within the bundle, so a
copied bundle does not depend on the original experiment directory.

See [EEGDiffuser details](eegdiffuser/README.md) for model behavior and
[graph loading and classifier training](../datacustom/README.md) for the next stage.
