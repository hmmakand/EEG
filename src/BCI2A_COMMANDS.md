# BCI IV 2a, all 9 subjects: training commands for version 27 and version 28

BCI Competition IV 2a, left hand (0) against right hand (1), 22 EEG electrodes,
one sample per trial (the 2 s after the cue). From now on the cohort is **all
nine subjects, 1 to 9**. Earlier runs left subject 4 out. Every command below
uses the new default, so `--subjects` is not needed.

- **Version 27** (`src.ourexpv27`) is **our method, latest implementation**.
- **Version 28** (`src.ourexpv28reproduce`) holds the **comparison papers**.
  They run on version 27's subjects, samples and test folds, so their numbers
  can be compared with ours row by row.

Seed 12345 for every run, one run per configuration. If a result file of the
same name exists, a time stamp is added to the new one, so nothing is
overwritten. That also applies to the old 8-subject results, so look at the
`subjects` field in the JSON to tell the two apart.

Two protocols:

| Protocol | Version 27 | Version 28 | Folds |
|---|---|---|---|
| Leave one subject out (cross-subject) | `train.run_loso` | `--protocol loso` (default) | 9, one per held-out subject |
| Within subject | `train.run_within` | `--protocol within` | 10 per subject, 90 in all |

## Setup (once in every new terminal or tmux session)

Run from the EEG repository root.

```bash
cd /root/autodl-tmp/Experiment/EEG
BCI=/root/autodl-tmp/Experiment/EEG/data/bci-competition-iv-data-sets-2a   # A01T.mat ... A09T.mat
OUT27=/root/autodl-tmp/Experiment/EEG/outputs/ourexpv27
OUT28=/root/autodl-tmp/Experiment/EEG/outputs/ourexpv28reproduce
mkdir -p "$OUT27/logs" "$OUT28/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`A04T.mat` must be in `$BCI` now. The command stops with a message naming any
missing file.

## Version 27: our method

```bash
# Leave one subject out (the main result)
python -u -m src.ourexpv27.train.run_loso   --quiet --dataset bci2a --data-dir "$BCI" --output-dir "$OUT27" 2>&1 | tee "$OUT27/logs/v27_bci2a_loso.log"

# Within subject
python -u -m src.ourexpv27.train.run_within --quiet --dataset bci2a --data-dir "$BCI" --output-dir "$OUT27" 2>&1 | tee "$OUT27/logs/v27_bci2a_within.log"

# Figures, after the runs
python -m src.ourexpv27.analysis.figures "$OUT27"/bci2a/Version27_bci2a_*_seed12345*.json
```

Results go to `$OUT27/bci2a/`.

## Version 28: comparison papers

Every paper has the same entry point, `src.ourexpv28reproduce.run`. Its
results go to `$OUT28/<paper>/bci2a/`, as JSON and as a `.summary.txt` table.

### One paper at a time

```bash
# EEGNet
python -u -m src.ourexpv28reproduce.run --paper eegnet          --dataset bci2a --protocol loso   --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/eegnet_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper eegnet          --dataset bci2a --protocol within --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/eegnet_bci2a_within.log"

# M-ShallowConvNet
python -u -m src.ourexpv28reproduce.run --paper mshallowconvnet --dataset bci2a --protocol loso   --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/mshallowconvnet_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper mshallowconvnet --dataset bci2a --protocol within --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/mshallowconvnet_bci2a_within.log"

# EEG Conformer
python -u -m src.ourexpv28reproduce.run --paper conformer       --dataset bci2a --protocol loso   --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/conformer_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper conformer       --dataset bci2a --protocol within --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/conformer_bci2a_within.log"

# MSCFormer (the slowest: about a day and a half for leave one subject out)
python -u -m src.ourexpv28reproduce.run --paper mscformer       --dataset bci2a --protocol loso   --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/mscformer_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper mscformer       --dataset bci2a --protocol within --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/mscformer_bci2a_within.log"

# DB-Conformer
python -u -m src.ourexpv28reproduce.run --paper dbconformer     --dataset bci2a --protocol loso   --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/dbconformer_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper dbconformer     --dataset bci2a --protocol within --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/dbconformer_bci2a_within.log"

# SlimSeiz
python -u -m src.ourexpv28reproduce.run --paper slimseiz        --dataset bci2a --protocol loso   --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/slimseiz_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper slimseiz        --dataset bci2a --protocol within --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/slimseiz_bci2a_within.log"

# UADAAN
python -u -m src.ourexpv28reproduce.run --paper uadaan          --dataset bci2a --protocol loso   --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/uadaan_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper uadaan          --dataset bci2a --protocol within --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/uadaan_bci2a_within.log"

# PA-TCNet (version 27's cohort is its default; written out here to be explicit)
python -u -m src.ourexpv28reproduce.run --paper patcnet         --dataset bci2a --protocol loso   --cohort version27 --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/patcnet_bci2a_loso.log"
python -u -m src.ourexpv28reproduce.run --paper patcnet         --dataset bci2a --protocol within --cohort version27 --quiet --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/patcnet_bci2a_within.log"
```

### All papers in one go

The papers run one after another. This takes days (MSCFormer alone is more
than a day), so start it inside `tmux`.

```bash
for paper in eegnet mshallowconvnet conformer mscformer dbconformer slimseiz uadaan patcnet; do
  for protocol in loso within; do
    python -u -m src.ourexpv28reproduce.run --paper "$paper" --dataset bci2a --protocol "$protocol" --quiet \
      --data-dir "$BCI" --output-dir "$OUT28" 2>&1 | tee "$OUT28/logs/${paper}_bci2a_${protocol}.log"
  done
done
```

### Figures, after the runs

```bash
for paper in eegnet mshallowconvnet conformer mscformer dbconformer slimseiz uadaan patcnet; do
  python -m src.ourexpv28reproduce.evaluation.figures "$OUT28/$paper"/bci2a/*.json
done
```

## Options

- `--quiet` hides the progress bars. Leave it out to watch the training.
- `--save-models` keeps the trained models next to the result. It works for
  version 27 and for every version 28 paper except `patcnet`.
- `--epochs N` shortens a run, for a quick check before the real one (every
  version 28 paper and version 27's `run_within`; version 27's `run_loso` has
  no such option). Do not report numbers from shortened runs.
- `--subjects 4` (or any list) runs only those held-out subjects.
- Each paper's own options: `python -m src.ourexpv28reproduce.run --paper <paper> --help`.
