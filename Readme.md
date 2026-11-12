# GT-PGTA

GT-PGTA is a graph-temporal framework for EEG motor imagery decoding. The implementation is in [`src/GTPGTA`](src/GTPGTA) and supports leave-one-subject-out (LOSO) and within-subject experiments.

## Short abstract

Learning motor imagery patterns that generalise across stroke participants is difficult when individual calibration labels are limited. GT-PGTA combines EEG connectivity, temporal signals, and explicit signal descriptors in a three-branch classifier. Its target adaptation procedure screens pseudo-labels using prediction confidence and source-derived motor-region patterns, preserves accepted supervision, and adjusts participant-level score bias. The reported transductive LOSO balanced accuracies are 72.15% on Liu2024, suggesting that complementary signal representations and access to unlabelled target EEG can support cross-participant decoding.

## Method

- **Graph branch:** represents electrodes and their relationships using subband power, phase-locking, and covariance features.
- **Temporal branch:** extracts patterns from aligned EEG with shallow convolutions.
- **Wide branch:** maps explicit EEG descriptors directly to class scores through a linear layer.
- **Target adaptation:** filters pseudo-labels by confidence and motor-region template agreement, retains accepted labels, and centres target-participant logits at inference.

## Installation

Use Python 3.10 or newer.

```bash
git clone https://github.com/hmmakand/EEG.git
cd EEG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r src/GTPGTA/requirements.txt
```

The requirements include PyTorch, PyTorch Geometric, NumPy, SciPy, scikit-learn, tqdm, and MNE. For GPU execution, the environment must have a CUDA-compatible PyTorch installation. Use `--device cpu` to run without CUDA or `--device auto` to select the available device.

For subsequent sessions, activate the existing environment:

```bash
source .venv/bin/activate
```

## Dataset preparation

Obtain the datasets separately and replace the example paths below with your local directories. Recordings are not included in this repository.

| Dataset argument | Classification task | Expected layout under `--data-dir` |
|---|---|---|
| `liu2024` | Motor imagery versus rest | `edffile/sub-01/eeg/sub-01_task-motor-imagery_eeg.edf`, etc.; a parent directory containing `files/edffile/` is also accepted |
| `jia2019` | Left versus right motor imagery | `exp1-S1-left.mat`, `exp1-S1-right.mat`, etc. |
| `bci2a` | Left versus right motor imagery | `A01T.mat` through `A09T.mat` |

## Running experiments

Execute all commands from the repository root with the environment activated.

### Liu2024: leave-one-subject-out

```bash
python -u -m src.GTPGTA.train.run_loso \
  --dataset liu2024 \
  --data-dir /path/to/MNE-liu2024-data \
  --output-dir outputs/GTPGTA \
  --epochs 100 --patience 15 --seed 12345 \
  --device auto --save-models
```

### Liu2024: within-subject

```bash
python -u -m src.GTPGTA.train.run_within \
  --dataset liu2024 \
  --data-dir /path/to/MNE-liu2024-data \
  --output-dir outputs/GTPGTA \
  --epochs 100 --patience 15 --seed 12345 \
  --folds 10 --split-seed 42 \
  --device auto --save-models
```

For Jia2019 or BCI2a, change `--dataset` to `jia2019` or `bci2a` and supply the matching data directory. Add `--subjects 1` to evaluate only subject 1; LOSO still requires the other participants as source data. Add `--quiet` to hide progress bars.

### Run all datasets and both protocols

The wrapper runs six experiments sequentially and saves final model checkpoints:

```bash
DEVICE=auto bash src/GTPGTA/run_all.sh \
  /path/to/MNE-liu2024-data \
  /path/to/bci2a \
  /path/to/jia2019 \
  outputs/GTPGTA
```

The wrapper defaults to CUDA when `DEVICE` is unset. Set `DEVICE=cpu` for CPU execution.

## Reported Liu2024 results

Results are reported as mean ± standard deviation across participants. Bold indicates the best mean and underline the second-best mean in each column.

### LOSO

| Model | Balanced accuracy (%) | Precision (%) | Macro F1 (%) | Cohen's κ | AUC |
|---|---:|---:|---:|---:|---:|
| Conformer | 52.74 ± 6.84 | 45.47 ± 22.32 | 40.44 ± 11.46 | 0.055 ± 0.137 | 0.598 ± 0.191 |
| DBConformer | 67.36 ± 9.17 | 69.87 ± 9.31 | 66.09 ± 10.35 | 0.347 ± 0.183 | 0.760 ± 0.115 |
| EEGNet | <u>68.05 ± 10.24</u> | <u>70.09 ± 10.44</u> | <u>66.65 ± 11.50</u> | <u>0.361 ± 0.205</u> | <u>0.766 ± 0.113</u> |
| MSCFormer | 59.02 ± 7.19 | 60.78 ± 7.86 | 56.83 ± 8.45 | 0.180 ± 0.144 | 0.640 ± 0.100 |
| MShallowConvNet | 61.11 ± 9.38 | 63.97 ± 11.74 | 58.33 ± 11.17 | 0.222 ± 0.188 | 0.701 ± 0.133 |
| PA-TCNet | 61.48 ± 12.16 | 57.78 ± 20.69 | 56.09 ± 17.68 | 0.230 ± 0.243 | 0.717 ± 0.148 |
| SlimSeiz | 60.64 ± 8.27 | 62.69 ± 9.51 | 58.65 ± 9.75 | 0.213 ± 0.165 | 0.671 ± 0.112 |
| GT-PGTA (ours) | **72.15 ± 9.49** | **73.08 ± 10.05** | **71.93 ± 9.49** | **0.443 ± 0.190** | **0.787 ± 0.120** |

### Within-subject

| Model | Balanced accuracy (%) | Precision (%) | Macro F1 (%) | Cohen's κ | AUC |
|---|---:|---:|---:|---:|---:|
| Conformer | <u>75.63 ± 11.96</u> | <u>76.09 ± 12.05</u> | <u>75.51 ± 12.01</u> | <u>0.513 ± 0.239</u> | 0.816 ± 0.121 |
| DBConformer | 72.35 ± 11.32 | 72.87 ± 11.33 | 72.11 ± 11.50 | 0.447 ± 0.226 | 0.783 ± 0.118 |
| EEGNet | 75.37 ± 12.11 | 75.86 ± 12.17 | 75.21 ± 12.24 | 0.507 ± 0.242 | <u>0.817 ± 0.133</u> |
| MSCFormer | 72.32 ± 13.78 | 72.52 ± 13.82 | 72.22 ± 13.85 | 0.446 ± 0.276 | 0.754 ± 0.146 |
| MShallowConvNet | 73.95 ± 14.50 | 74.13 ± 14.51 | 73.87 ± 14.55 | 0.479 ± 0.290 | 0.792 ± 0.158 |
| PA-TCNet | 74.50 ± 13.25 | 74.87 ± 14.41 | 73.69 ± 14.89 | 0.490 ± 0.265 | 0.796 ± 0.145 |
| SlimSeiz | 67.33 ± 13.04 | 67.54 ± 13.11 | 67.16 ± 13.11 | 0.347 ± 0.261 | 0.707 ± 0.149 |
| GT-PGTA (ours) | **77.92 ± 12.36** | **78.07 ± 12.39** | **77.88 ± 12.40** | **0.558 ± 0.247** | **0.839 ± 0.131** |
