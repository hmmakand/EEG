# DBConformer: reproduction on our datasets

Wang Z., Wang H., Jia T., He X., Li S., Wu D. (2026). "DBConformer:
Dual-Branch Convolutional Transformer for EEG Decoding". IEEE Journal of
Biomedical and Health Informatics 30(5), 4134-4147. Paper:
`src/temp/DBConformer_Dual-Branch_Convolutional_Transformer_for_EEG_Decoding.pdf`.
Code: https://github.com/wzwvv/DBConformer, copied unchanged to
[original/](original/README.md) (PolyForm Noncommercial licence, per its README).

Purpose: a baseline for our method, like the other papers here. It is trained
and tested on **exactly the subjects, tasks, samples and test folds of version
27**, with both of version 27's protocols.

Status: **ready to run**; only short trial runs so far.

## How to run

```bash
cd /workspaces/EEG && source .venv/bin/activate
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv28reproduce/outputs
python -u -m src.ourexpv28reproduce.run --paper dbconformer --dataset bci2a --protocol loso --quiet --save-models \
  --data-dir /workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a --output-dir "$OUT"
```

`--dataset {liu2024,bci2a,jia2019}`, `--protocol {loso,within}`,
`--subjects`, `--epochs` (default 100), `--seed` (default 1),
`--save-models` (the model after the last epoch of every fold). The result is
`<output-dir>/dbconformer/<dataset>/DBConformer_<dataset>_<LOSO|Within>_seed1.json`
with a `.summary.txt`; figures with
`python -m src.ourexpv28reproduce.evaluation.figures <result>.json`.

Run time on a GTX 1060: about 34 s per bci2a leave-one-subject-out fold
(about 5 minutes for the run) and 3.5 s per liu2024 within-subject fold
(about half an hour for the run). This is the cheapest of the transformer
baselines: one model, 100 epochs.

## The two scores of every run

| Score | How it is chosen | Uses the test labels |
|---|---|---|
| **final epoch** (main) | After epoch 100. This is what the paper's code reports, and version 27's protocol | No |
| **highest test epoch** | The epoch with the highest test accuracy | Yes; for comparison only |

As for the other papers, every test sample's prediction and probability (ROC
curves), the training and test curves, the time, memory and model size are
recorded.

## 1. What the paper reports

Chronological-order, cross-validation (5 folds, within subject) and
leave-one-subject-out results on six motor-imagery datasets (Table IV, an
image in our copy of the PDF), five seeds (1 to 5) averaged. Its
BNCI2014001 is BCI IV 2a, left against right hand, first session, all 9
subjects, 4 s from the cue: the closest of all papers here to our bci2a
(all 9 subjects, 2 s from the cue). The paper's numbers are not
copied here; compare with the published table.

## 2. The specification, and where each part comes from

| Part | Value | Source |
|---|---|---|
| Temporal branch | Conv1d 1x1 (channels to 40) + BN; depthwise Conv1d kernel 63 + BN; GELU; average pooling 125 (0.5 s) into patches; dropout 0.5; learned positions; transformer; mean over patches | Section III-B2, DBConformer.py |
| Spatial branch | Per channel Conv1d (1 to 16, kernel 25, stride 5), ELU, mean over time, Linear to 40; learned positions; transformer; channel attention (Linear, tanh, Linear, softmax over channels) | Section III-B3, DBConformer.py |
| Transformers | Pre-LayerNorm, 10 heads, feed-forward 160, dropout 0.5; depth 2 for bci2a, 6 for liu2024 and jia2019 (leave one subject out); 2 for all within subject | DBConformer_LOSO.py (2 for BNCI2014001/2014004, 6 for its other datasets), DBConformer_CV.py (2) |
| Classifier | [temporal; spatial] (80) to 64, ELU, dropout 0.5, to 32, ELU, dropout 0.3, to classes | Section III-B4, code |
| Pre-processing | MOABB's standard motor-imagery pre-processing: band-pass 8-32 Hz (4th-order Butterworth, forward and backward) | Section IV-A4 ("the standard preprocessing steps in MOABB") |
| Alignment | Euclidean alignment per training subject; online alignment of the test trials (each with the mean covariance of the test trials up to it) | Section IV-A4, utils.py, alg_utils.py |
| Training | Adam, learning rate 0.001, batch 32, 100 epochs, cross-entropy; the last epoch is the result | Section IV-C3, scripts |
| Seed | 1 (the paper averages seeds 1 to 5) | Paper |

The port computes the same function as the published model: with the
published weights copied in, the outputs are identical (tested). The
alignment gives the same values as the code's `EA` and online alignment
(tested).

## 3. Where the code and the paper differ, and what was done

- **Test alignment within subject:** the code's CV script aligns the test fold
  offline and then online a second time; the paper describes online
  alignment of the test trials. Here the test samples are aligned online
  once, in both protocols.
- **The last time point:** the code drops one sample before pooling (its
  trials have 1,001 samples) except for its 750-sample dataset. Our samples
  are not one sample too long, so nothing is dropped.
- **Seeds:** one run with seed 1 instead of the average of five.

## 4. What is ours, not the paper's

| | Here (version 27's) | Paper |
|---|---|---|
| Task | liu2024: imagery against rest; bci2a and jia2019: left against right hand | Left against right hand (and others) |
| Windows | liu2024: 1.9 s (3 patches); bci2a: 0-2 s after the cue (4 patches); jia2019: the whole 6.8 s epoch with a common average reference (13 patches) | BNCI2014001: 4 s (8 patches) |
| Within subject | Version 27's 10 folds | 5 chronological folds |
| Rank-deficient covariances | jia2019 (common average reference, and subject 6's 62 channels): the alignment leaves out the missing directions | Not needed in the paper's data |

The alignment uses the test subject's samples without their labels, as our
method's alignment does.

## 5. Files

```
settings.py      model, depth per dataset and protocol, preparation and training settings
preparation.py   version 27's samples, band-passed; Euclidean alignment, offline and online
network.py       DBConformer, ported from original/models/DBConformer.py
training.py      one fold: 100 epochs, the two scores, curves, time, memory
experiment.py    leave one subject out and within subject, with the alignment of each
command.py       the command-line options and the result name
original/        the published code, unchanged
```
