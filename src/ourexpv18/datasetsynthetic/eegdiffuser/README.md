# EEGDiffuser for prepared signal folds

This local implementation is used by the v18 fold-first generation runner.
It reads each prepared job's permitted original trials. It imports no code from
another experiment or `temp`, and does not create graphs.

Use the public runner to retain protocol isolation and final-manifest generation:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv18/requirements.txt
python -m pip install -r src/ourexpv18/datasetsynthetic/eegdiffuser/requirements.txt

# 1. Prepare original signals and fold assignments
python -m src.ourexpv18.datasetsynthetic.prepare \
  --data-dir data/kaggle/bci-competition-iv-data-sets-2a \
  --output src/ourexpv18/datasyn/fold_first_self_conditioning

# 2. Train diffusion models and generate synthetic signals
python -u -m src.ourexpv18.datasetsynthetic.generate \
  --bundle src/ourexpv18/datasyn/fold_first_self_conditioning \
  --protocol within_subject \
  --epochs 100 \
  --device cuda
```

For an interrupted run, repeat with `--resume` and the same settings. Completed
jobs are verified and skipped. Use `--protocol cross_subject` for the separate
cross-subject generators. For a fresh experiment, prepare a new bundle first.
See [preparation and protocol rules](../README.md).

Within-subject diffusion has 119–120 training and ten internal validation trials
per model. Cross-subject preparation uses subject-specific generators with 134
training and ten validation originals each. Internal validation selects the
checkpoint and never updates weights. The source trial indices are saved and
verified; signal paths resolve relatively within the copied bundle.

From a standalone folder, use `python datasetsynthetic/generate.py` with the
same options. The lower-level `eegdiffuser/train.py` is an implementation helper;
its input must be a prepared job input directory, not the unpartitioned original
signal directory. The public runner supplies this automatically.

## Reference behavior and explicit adaptations

The model and diffusion implementation were copied from EEGDiffuser by Jiquan
Wang et al.; the supplied license is retained. Source reference:
`https://github.com/wjq-learning/EEGDiffuser`. No source download is performed.

All defaults live in `config.py`:

- One independently initialized model per prepared generation job; seed 8888.
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

## Checkpoints and outputs

Every job writes its own configuration, metadata, `internal_partitions.csv`,
training history, `checkpoints/subject_XX/best.pt`, and `latest.pt`. The best
ordinary model is selected by internal validation loss. Latest checkpoints
include optimizer, scheduler, EMA, RNG state, and the matching best model snapshot
for reproducible epoch-boundary recovery. They are saved every ten epochs and
at the end. Resume requires matching settings and source hashes.

Each generated NPZ contains `signals` shaped `[100, 1000, 22]` in original scale,
`labels`, `subject_ids`, and `synthetic_trial_ids`. The manifest records class,
subject, array index, generation seed, and selected checkpoint hash and epoch.
The protocol manifest namespaces synthetic identities by their generation job.

The default sampling procedure uses 50 trials per class, classifier-free guidance
4.0, 1,000 reverse diffusion steps, no clipping, and generation batch size one.
Generation seed derives from base seed 8888 and subject ID. Successful generation
does not by itself establish physiological quality or classification improvement.

## Verify

```bash
python -m unittest discover \
  -s src/ourexpv18/datasetsynthetic/eegdiffuser/tests -v
```

Tests use small CPU models to verify isolation, scaling, checkpoint selection,
resume, and reproducible generation.

## Inherited global-fusion architecture and fast verification

The active model is `models/eegdiffuser.py` with global-token fusion.
`Config.model_architecture` is `eegdiffuser_global_fusion_v1` and
`Config.num_global_tokens` defaults to 4. Both fields are saved with each run;
the token count is passed explicitly to the model constructor. Saved runs without
this identity, with incompatible settings, or with a different model-source hash
are rejected before reuse, resume, or generation finalization. Copied v14 outputs
remain historical data and cannot count as generation by this model. Prepare a
fresh bundle for new model results; do not relabel or edit old run metadata.

Fast verification from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover \
  -s src/ourexpv18/datasetsynthetic/tests -v
```

These checks use temporary CPU fixtures: two channels, ten time samples, embedding
size 8, one transformer/fusion block, and two global tokens. One generation job
uses two training trials in one batch, two validation trials, and produces two
signals using 21 diffusion steps. A separate three-update gradient check verifies
that the initially zero-gated fusion branch starts learning. No real folds are
trained and no saved EEG data is modified. These tests verify functionality and
isolation, not signal quality or downstream classifier improvement.
