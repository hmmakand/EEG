# SlimSeiz: reproduction on our datasets

Lu G., Peng J., Huang B., Gao C., Stefanov T., Hao Y., Chen Q. (2025).
"SlimSeiz: Efficient Channel-Adaptive Seizure Prediction Using a
Mamba-Enhanced Network". IEEE International Symposium on Circuits and Systems
(ISCAS), DOI 10.1109/ISCAS56072.2025.11043364. Paper:
`src/temp/SlimSeiz_Efficient_Channel-Adaptive_Seizure_Prediction_Using_a_Mamba-Enhanced_Network.pdf`.
Code: https://github.com/guoruilu/SlimSeiz (Jupyter notebooks), copied
unchanged to [original/](original/README.md).

Purpose: a baseline for our method. **SlimSeiz was designed for seizure
prediction** (pre-ictal against inter-ictal, 4 s segments of raw scalp EEG),
not for motor imagery; it is applied here, unchanged, to our tasks, as the
DBConformer benchmark also does. It is trained and tested on **exactly the
subjects, tasks, samples and test folds of version 27**, with both of version
27's protocols.

Status: **ready to run**; only short trial runs so far.

## How to run

```bash
cd /workspaces/EEG && source .venv/bin/activate
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv28reproduce/outputs
python -u -m src.ourexpv28reproduce.run --paper slimseiz --dataset bci2a --protocol loso --quiet --save-models \
  --data-dir /workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a --output-dir "$OUT"
```

`--dataset {liu2024,bci2a,jia2019}`, `--protocol {loso,within}`,
`--subjects`, `--epochs` (default 30), `--channels` (default 8), `--seed`
(default 12345), `--save-models` (the validation-selected model of every
fold, with its selected channels). The result is
`<output-dir>/slimseiz/<dataset>/SlimSeiz_<dataset>_<LOSO|Within>_seed12345.json`
with a `.summary.txt`; figures with
`python -m src.ourexpv28reproduce.evaluation.figures <result>.json`.

### Run time

The network trains in seconds; the channel selection (30 repetitions x every
channel x PCA, SMOTE and a decision tree) takes most of the time and runs on
the CPU. On this machine (12 CPU threads): about 40 s per bci2a
within-subject fold (7.5 min per subject) and 5.5 min per bci2a
leave-one-subject-out fold. Estimates for whole runs: bci2a about 1 h per
protocol; liu2024 about 6 h within subject and 8 h leave one subject out;
jia2019 (63 channels) several hours per protocol. The time per subject is
printed as each finishes.

## The three scores of every run

| Score | How it is chosen | Uses the test labels |
|---|---|---|
| **validation-selected** (main) | The code's protocol: the epoch with the highest validation accuracy over 30 epochs | No |
| **final epoch** | After epoch 30 | No |
| **highest test epoch** | The epoch with the highest test accuracy | Yes; for comparison only |

Every fold also records the selected channels and how often each channel
was among the best k (`selected_channels`, `channel_counts`; indices into
the dataset's channel order), with the usual predictions, probabilities,
curves, time, memory and model size.

## 1. What the paper reports

Seizure prediction on CHB-MIT (24 cases, 8 selected channels): accuracy
94.8%, sensitivity 95.5%, specificity 94.0%, 10-fold cross-validation, 21.2 K
parameters (Table II); SRH-LEI (8 patients): 92.7% (Table III). Not
comparable with motor imagery.

## 2. The specification, and where each part comes from

| Part | Value | Source |
|---|---|---|
| Channel selection | Per channel: 70/30 random split, down-sampled by 2, PCA (60 components), SMOTE on the training part, decision tree, test accuracy; the best k channels; 30 repetitions; the k channels most often among the best | Section II-B, Loop_select_ch_PCA_SMOTE_DT.ipynb |
| k | 8 | Section III-B (the training notebook uses 6) |
| Network | Conv1d 21 (16) + max pool 8; residual block (1x1 + [11, 3]) (16) + max pool 4; residual block, stride 2 (1x1 + [5, 3]) (32); Mamba block (expansion 2, state 16, convolution 3) with RMSNorm and residual; global average pooling; Linear 2 | Section II-C, Fig. 2, OneDCNN in the notebook |
| Input | Raw signal in volts x 10^4; no filter, no normalisation | Notebook (`data * 1e4`) |
| Loss | 0.5 x cross-entropy + 0.5 x supervised contrastive (temperature 0.08) on the 32 pooled features | Section II-C3, notebook |
| Training | Adam, learning rate 0.001, weight decay 0.0001, batch 32, 30 epochs | Notebook |
| Validation | 20 % of the training side, at random; the epoch with the highest validation accuracy is tested | Notebook |
| Folds | 10-fold (stratified) cross-validation within patient | Section III, notebook |
| Seed | 12345 | The code sets none; ours |

The port computes the same function as the notebook's model: with its
weights copied in, the outputs and features are identical (tested). The
contrastive loss is the notebook's: it passes the rows of the scaled
similarity matrix as embeddings to pytorch-metric-learning's NTXentLoss; that
loss is re-implemented here (no extra package) and gives the library's values
(checked). With 8 input channels the network has 22,946 parameters; the paper
reports 21.2 K.

## 3. Where the code and the paper differ, and what was done

- **Number of channels:** the paper selects 8; the training notebook uses 6.
  The paper's 8 is used (`--channels` changes it).
- **Channel selection data:** the notebook selects channels on all of a
  patient's data, test segments included. Here only the training side of each
  fold is used, so the test labels never influence the selection.
- **Validation split:** the notebook draws 20 % of the training indices
  without a seed; here it is seeded, and whole trials (liu2024's imagery and
  rest samples of one trial) stay together.

## 4. What is ours, not the paper's

| | Here (version 27's) | Paper |
|---|---|---|
| Task | liu2024: imagery against rest; bci2a and jia2019: left against right hand | Pre-ictal against inter-ictal |
| Samples | liu2024 1.9 s (7 time steps for the Mamba block), bci2a 2 s (8), jia2019 6.8 s (27), at 250 Hz | 4 s at 256 Hz (16) |
| Channels to select from | liu2024 29, bci2a 22, jia2019 63 | 17 to 22 bipolar channels |
| Protocol | Within subject with version 27's 10 folds; also leave one subject out, with 20 % of the training subjects for validation | Within patient only |
| Scale | Volts x 10^4; the BCI IV 2a reader returns microvolts, which are converted to volts first | Volts x 10^4 |

## 5. Files

```
settings.py      channel selection and training settings
preparation.py   version 27's samples, raw, volts x 1e4
channels.py      the channel selection: PCA, SMOTE, decision tree, 30 repetitions
network.py       the convolution + Mamba network, ported from the notebook
training.py      one fold: the contrastive loss, validation selection, the three scores
experiment.py    leave one subject out and within subject: channel selection, validation split, training
command.py       the command-line options and the result name
original/        the published notebooks, unchanged
```
