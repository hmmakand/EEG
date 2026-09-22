# EEGDiffuser on original within-subject EEG

This package trains a separate label-conditioned diffusion model for each subject,
then generates synthetic EEG for downstream training. It uses only the **training**
assignments saved under `../../datasyn/original/`. There are no runtime imports
from `temp`, no graph creation, and no changes to version ten or original data.

## Full training and generation

From the repository root, with the existing environment activated:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexperimentversionthirteen/datasetsynthetic/eegdiffuser/requirements.txt
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="src/ourexperimentversionthirteen/datasyn/synthetic/eegdiffuser_${run_stamp}"
python -u -m src.ourexperimentversionthirteen.datasetsynthetic.eegdiffuser.train \
  --original-dir src/ourexperimentversionthirteen/datasyn/original \
  --run-dir "$run_dir" \
  --subjects 1 2 3 5 6 7 8 9 \
  --epochs 1000 --device cuda --samples-per-class 50 --generate
```

`--generate` generates each subject's synthetic data after its training finishes.
Omit it to train only. To generate from a fully trained run later:

```bash
python -m src.ourexperimentversionthirteen.datasetsynthetic.eegdiffuser.generate \
  --run-dir "$run_dir"
```

For background execution, create a unique run directory and capture its process ID:

```bash
mkdir -p "$run_dir"
nohup python -u -m src.ourexperimentversionthirteen.datasetsynthetic.eegdiffuser.train \
  --run-dir "$run_dir" --epochs 1000 --device cuda --generate \
  > "${run_dir}.log" 2>&1 &
echo $! > "${run_dir}.pid"
```

Do not run multiple writers for the same run directory. Run configuration is
immutable once saved; choose a new directory for different settings. With a
standalone copy of version thirteen, direct script execution also works:

```bash
python datasetsynthetic/eegdiffuser/train.py --epochs 1000 --device cuda --generate
```

## Internal validation without touching held-out original data

For each subject, the 100 original training trials are divided into:

| Subset | Left | Right | Total |
| --- | ---: | ---: | ---: |
| Diffusion training | 45 | 45 | 90 |
| Internal validation | 5 | 5 | 10 |

Both stages shuffle with local PCG64 streams derived from seed 42, subject ID,
stage, and class/subset. Assignments and within-subset order are persisted in
`internal_partitions.csv`. They retain original trial identities and the original
`partition=training` field. Resuming verifies and reuses them.

The NPZ adapter reads only selected NPY signal rows from the compressed archive.
ZIP can decompress skipped bytes internally, but original validation/test rows
are never materialized as EEG arrays, checked for signal values, or sent to the
model. Source artifacts are hashed as bytes for provenance. These hashes do not
use held-out data to fit or select anything.

## Reference behavior and explicit adaptations

The model and diffusion implementation were copied from EEGDiffuser by Jiquan
Wang et al.; the supplied license is retained. Source reference:
`https://github.com/wjq-learning/EEGDiffuser`. No source download is performed.

All defaults live in `config.py`:

- One independently initialized model per subject; seed 8888 for each subject.
- Input `[batch, 22, 1000]`, two labels, embedding dimension 512, patch size 5,
  four transformer blocks, 16 attention heads, MLP ratio 4, class dropout 0.1,
  and learned diffusion variance. Only dimensions/classes adapt the architecture.
- Signals are divided by 100 before float32 conversion. Generated signals are
  multiplied by 100 before saving. No fitted normalization, filtering, features,
  or graph creation is performed.
- 1,000 epochs, batch size 1, shuffled training loader, unshuffled validation,
  AdamW learning rate 1e-4 and weight decay 0.05. CosineAnnealingLR steps after
  every optimizer update with `T_max=epochs * training_batches` and `eta_min=1e-6`.
- Diffusion uses 1,000 steps, the linear schedule, and the reference combined
  noise-prediction/learned-variance loss. No classification loss, gradient clipping,
  early stopping, or mixed precision is added.
- EMA parameters are initialized from the ordinary model and updated with decay
  0.9999. The **ordinary model**, not EMA, is evaluated, selected, and sampled.
- Validation reproduces the reference batch-average loss. Timesteps are drawn
  with a CPU generator seeded by batch index; diffusion noise is freshly sampled.
- Best checkpoint selection requires a strict validation-loss improvement; the
  earliest epoch wins a tie. Validation never updates model weights.
- **Documented correction:** `model.train()` is restored at every epoch because
  the reference evaluator leaves the model in evaluation mode. This keeps
  classifier-free label dropout active after the first epoch.
- Reference CUDA seeding and `cudnn.deterministic=True` are retained. Exact GPU
  results across devices/library versions are not guaranteed.

GPU index defaults to the first GPU rather than the reference machine's GPU 3.
NPZ/CSV reading replaces FACED LMDB loading and hard-coded paths. The original
MATLAB recordings are not read again during diffusion training.

## Checkpoints and recovery

`best.pt` holds the ordinary model with lowest internal validation loss.
`latest.pt` holds model, EMA, optimizer, scheduler, RNG states, epoch, best-score
information, a matching best-model snapshot, history, and configuration. Latest is saved every 10 epochs and at
the end of training or an explicit pause, reducing writes of large checkpoints.
After an interruption, resume from that saved epoch; up to nine epochs may repeat.
The saved best-model snapshot also recovers a missing or newer partially written
`best.pt`, keeping checkpoint selection consistent with the resumed epoch.
Do not change epoch count when resuming: it changes the scheduler horizon.

```bash
python -u -m src.ourexperimentversionthirteen.datasetsynthetic.eegdiffuser.train \
  --run-dir "$run_dir" --resume --generate
```

For a controlled pause, use `--stop-after-epoch N` with the intended full
`--epochs` value. This preserves the full-run schedule. Source/config/partition
mismatches cause errors rather than replacing existing data. Only load trusted
locally produced resume checkpoints: they include Python/NumPy RNG state.

## Synthetic output

For each subject, generation requests 50 samples of each class (100 total),
regardless of the 90-trial diffusion training subset. It samples Gaussian noise
with classifier-free guidance 4.0, null label 2, all 1,000 reverse steps, and
`clip_denoised=False`. Generation seed is derived reproducibly from base seed
8888 and subject ID. Batch size defaults to 1, matching the reference setting.

```text
<run_name>/
├── config.json
├── metadata.json
├── internal_partitions.csv
├── manifest.csv
├── checkpoints/subject_01/{best,latest}.pt
├── history/subject_01.json
├── subject_01.npz
└── ...
```

Each NPZ contains `signals` (`[100, 1000, 22]`, float32, original scale),
`labels`, `subject_ids`, and unique `synthetic_trial_ids`. The manifest records
class, subject, array index, synthetic ID, generation seed, and checkpoint hash
and epoch. It makes no claim that a generated sample corresponds to a particular
original trial. Outputs are intended only for downstream training.

Metadata records source hashes, internal-manifest hash, code hashes, installed
versions, device, configuration, per-subject progress, and generated artifact
hashes. Generation reuses completed artifacts after verification. Original
validation and test partitions remain available for future classifier evaluation.
Successful generation alone does not establish physiological quality or improved
classification performance.

## Smoke test and automated checks

```bash
python -m unittest discover \
  -s src/ourexperimentversionthirteen/datasetsynthetic/eegdiffuser/tests -v
python -u -m src.ourexperimentversionthirteen.datasetsynthetic.eegdiffuser.train \
  --run-dir src/ourexperimentversionthirteen/datasyn/synthetic/smoke_run \
  --subjects 1 --epochs 1 --samples-per-class 1 --device cuda --generate
```

The short real-data run creates only two diagnostic synthetic trials. It is not
part of the full 800-trial dataset and is not a trained scientific baseline.
Tests cover data isolation, class balance, scaling, model mode restoration,
checkpoint selection, exact CPU resume, generation reproducibility, and artifact
reuse. Tiny CPU models are used in tests to keep them fast.
