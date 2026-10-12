# V17: prepare folds, then train and generate EEG

Run these commands from the repository root. V17 uses **DiT with self-conditioning**,
with no global fusion. The command below states this explicitly with
`--self-conditioning`; v17 also enables it by default when the flag is omitted.

Activate the environment and install dependencies once:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv17/requirements.txt
python -m pip install -r src/ourexpv17/datasetsynthetic/eegdiffuser/requirements.txt
```

## 1. Prepare the original signals and folds

```bash
python -m src.ourexpv17.datasetsynthetic.prepare \
  --data-dir data/kaggle/bci-competition-iv-data-sets-2a \
  --output src/ourexpv17/datasyn/fold_first_self_conditioning
```

Change `--data-dir` if your original MATLAB files are elsewhere. This saves the
original signals and fold assignments; it does not train a model or create graphs.
The new output directory keeps this experiment separate from copied old results.

## 2. Train and generate synthetic signals

```bash
python -u -m src.ourexpv17.datasetsynthetic.generate \
  --bundle src/ourexpv17/datasyn/fold_first_self_conditioning \
  --protocol within_subject \
  --epochs 100 \
  --device cuda \
  --self-conditioning
```

This trains one model per subject and fold: **8 subjects × 10 folds = 80 models**,
each for 100 epochs. Each model generates **50 left + 50 right = 100 synthetic
trials**, using its best internal-validation checkpoint. Held-out fold trials
are excluded from diffusion training and checkpoint selection.

Signals, checkpoints, and training histories are saved under:

```text
src/ourexpv17/datasyn/fold_first_self_conditioning/
```

After all within-subject jobs finish, `protocols/within_subject.csv` is ready for
[graph loading and classifier training](../../datacustom/README.md). Pass this
same bundle path to the classifier.

## Batch size and resuming

Set batch sizes in [config.py](config.py) before starting a fresh run:

```python
batch_size: int = 32       # Training and validation; repository default is 1
sample_batch_size: int = 1 # Synthetic generation, controlled separately
```

The command above does not override these settings. Batch size 32 has not been
verified with the full model on your GPU.

If interrupted, rerun step 2 with `--resume` added. Keep the same epoch count and
configuration. Completed jobs are verified and skipped. Do not resume copied
checkpoints from the old architecture; use the fresh bundle above.
