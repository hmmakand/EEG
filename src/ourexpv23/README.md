# Version twenty-three: the best left/right EEG classifier, isolated

Version twenty-three contains only the code that produces our best result for
left-hand versus right-hand motor imagery on BCI Competition IV 2a. Everything
that was tried and did not help in versions 10 to 22 has been left out of the
code and is recorded in this file instead.

The code runs two experiments: leave one subject out (LOSO), which is the
main result, and within-subject cross-validation.

**Result: 0.790 mean accuracy over five seeds** (range 0.781 to 0.800) in
leave-one-subject-out evaluation: train on seven subjects, test on an eighth
that the model has never seen. The classical reference (CSP + LDA with
alignment) scores 0.77 on the same trials. Within subject the same model
scores 0.800 (one seed); see [Within-subject experiment](#within-subject-experiment).

Contents of this file:

1. [Run it](#1-run-it)
2. [Code map](#2-code-map)
3. [The pipeline, step by step](#3-the-pipeline-step-by-step)
4. [Results](#4-results)
5. [What works, and the evidence](#5-what-works-and-the-evidence)
6. [What does not work](#6-what-does-not-work)
7. [How to read and report the numbers](#7-how-to-read-and-report-the-numbers)
8. [Ideas not tested](#8-ideas-not-tested)
9. [Where this version comes from](#9-where-this-version-comes-from)

## 1. Run it

Use Python 3.10 or newer. From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r src/ourexpv23/requirements.txt

RAW=/workspaces/datadisk/databci/eegdata/kaggle/bci-competition-iv-data-sets-2a
OUT=/workspaces/datadisk/databci/bciupperlimbactionexperiment/ourexpv23/outputs

python -u -m src.ourexpv23.train.run_loso \
  --data-dir "$RAW" \
  --output "$OUT/Version23_LOSO_seed12345.json" 2>&1 | tee "$OUT/Version23_LOSO_seed12345.log"
```

It reads `A01T.mat` to `A09T.mat` (subject 4 is not used), builds the graphs in
memory, trains and tests the eight folds, and writes one JSON file. The
whole run takes about 7 minutes on a GTX 1060.

| Option | Default | Meaning |
| --- | --- | --- |
| `--data-dir` | required | Folder with the `A0xT.mat` recordings |
| `--output` | timestamped file in `./outputs` | Result JSON |
| `--seed` | 12345 | Seed of the first model of each fold |
| `--subjects` | all configured | Held-out subjects to run, for example `--subjects 1 8` |
| `--all-subjects` | off | Use all nine subjects. The reported result leaves out subject 4, as earlier versions did; with this flag LOSO has nine folds of 1152 training trials. Not run yet |
| `--device` | `auto` | `auto`, `cpu` or `cuda` |
| `--quiet` | off | Hide the progress bars |

The within-subject experiment (about 40 minutes) uses the same options plus
`--folds`, `--split-seed` and `--epochs`:

```bash
python -u -m src.ourexpv23.train.run_within \
  --data-dir "$RAW" \
  --output "$OUT/Version23_Within_seed12345.json" 2>&1 | tee "$OUT/Version23_Within_seed12345.log"
```

Tests (CPU only, a few seconds, no real data needed):

```bash
python -m unittest discover -s src/ourexpv23/tests -t .
```

The code uses relative imports, so the folder can be copied anywhere as a
package and run with `python -m <package>.train.run_loso`.

## 2. Code map

About 1,200 lines in total, tests included (version 22 had about 8,100).

| File | Content |
| --- | --- |
| [data/config.py](data/config.py) | `DataConfig`: every rule that turns a recording into graphs |
| [data/recordings.py](data/recordings.py) | Read the MATLAB files and cut out the left/right trials |
| [data/signal.py](data/signal.py) | Band-pass filter, analysed window, per-subject Euclidean alignment |
| [data/graph.py](data/graph.py) | Band powers, phase-locking value, covariance; one trial to one graph |
| [model/config.py](model/config.py) | `ModelConfig`: the model sizes |
| [model/layers.py](model/layers.py) | `GatedGCN`, `PairAttention`, `GraphLayer`, `TemporalEncoder` |
| [model/model.py](model/model.py) | `EEGGraphModel`: graph head + wide path + temporal encoder |
| [train/config.py](train/config.py) | `TrainConfig`: schedule, optimiser, seeds, evaluation |
| [train/engine.py](train/engine.py) | One training epoch, the seed ensemble, evaluation with logit centring |
| [train/loso.py](train/loso.py) | The leave-one-subject-out loop |
| [train/within.py](train/within.py) | The within-subject cross-validation loop |
| [train/metrics.py](train/metrics.py) | Accuracy, balanced accuracy, macro F1, confusion matrix |
| [train/run_loso.py](train/run_loso.py), [train/run_within.py](train/run_within.py) | Command-line entry points |
| [tests/](tests/) | 15 tests on a small synthetic recording |

There are no alternative models, no optional paths and no switches for methods
that failed. Each configuration value is the one used for the result.

## 3. The pipeline, step by step

### 3.1 Trials ([data/recordings.py](data/recordings.py))

| Rule | Value |
| --- | --- |
| Subjects | 1, 2, 3, 5, 6, 7, 8, 9 by default; `--all-subjects` adds subject 4 |
| Sessions | Every session that contains trials: the six motor-imagery runs, at positions 3 to 8 (zero-based) for most subjects and 1 to 6 for subject 4 |
| Channels | Indices 0 to 21, the 22 EEG electrodes |
| Classes | Label 1 becomes 0 (left hand), label 2 becomes 1 (right hand); feet and tongue are skipped |
| Trial | 4 seconds from the trial start at 250 Hz: 1000 samples by 22 channels |
| Count | 144 trials per subject, 72 per class, in recording order |

The trial start is 2 seconds before the cue, so a trial holds 2 seconds of rest
and the first 2 seconds of imagery.

### 3.2 Signal ([data/signal.py](data/signal.py))

| Step | Rule |
| --- | --- |
| Align | Every trial of a subject is multiplied by the inverse square root of that subject's mean spatial covariance (Euclidean alignment). Afterwards the subject's mean covariance is the identity. No labels are used |
| Filter | Fifth-order Butterworth band-pass, zero-phase, 8 to 30 Hz, applied to the whole trial |
| Window | Seconds 2 to 4 of the filtered trial: 500 samples, the part after the cue |

### 3.3 Graph ([data/graph.py](data/graph.py))

One graph per trial; the 22 electrodes are the nodes.

| Field | Shape | Content |
| --- | --- | --- |
| `x` | `[22, 6]` | Band power of each electrode in 8-12, 12-16, 16-20, 20-24, 24-28, 28-32 Hz (Welch, 2-second Hann window, Simpson integration) |
| `edge_index` | `[2, 462]` | Every ordered electrode pair, no self-loops |
| `edge_attr` | `[462, 2]` | Phase-locking value and covariance of the pair in the window |
| `signal` | `[22, 500]` | The aligned, filtered, windowed signal of each electrode |
| `y` | scalar | 0 left hand, 1 right hand |

### 3.4 Model ([model/model.py](model/model.py))

The logits are the sum of three paths. 44,822 parameters.

| Path | What it does |
| --- | --- |
| **Temporal encoder** | Reads `signal`. Eight temporal filters of 25 samples shared by all electrodes, then 16 learned spatial filters across the electrodes, squaring, average pooling (75 samples, stride 15), log, dropout 0.7, and a linear layer. A spatial filter followed by log power is what CSP computes; here the filters are learned |
| **Graph head** | Node stream: log band powers plus a learned embedding per electrode. Pair stream: one vector per ordered electrode pair from the edge features, plus 8-step random-walk encodings. Two layers, each running a GatedGCN over the edges and GRIT attention over all pairs (hidden size 32, four heads, dropout 0.3, attention dropout 0.2, BatchNorm). Mean pooling of nodes and of pairs, then a two-layer MLP |
| **Wide path** | One linear layer on the flattened log band powers and edge features (dropout 0.5), so a specific electrode or pair can be weighted. Starts at zero |

The model assumes that every graph lists the same 22 electrodes in the same
order.

### 3.5 Training and evaluation ([train/](train/))

| Setting | Value |
| --- | --- |
| Protocol | Leave one subject out: eight folds, 1008 training trials and 144 test trials each |
| Epochs | 20, fixed; no validation set and no early stopping |
| Batch size | 32, shuffled |
| Optimiser | Adam, learning rate 0.001, weight decay 0.001 |
| Loss | Cross-entropy |
| Models per fold | 3, with seeds `seed`, `seed + 1`, `seed + 2`; their class probabilities are averaged |
| Logit centring | The held-out subject's mean logit is subtracted before the decision |
| Evaluation | Once, after the final epoch, on the held-out subject's 144 trials |

## 4. Results

### Five seeds

| Seed | 12345 | 100 | 200 | 300 | 400 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LOSO accuracy | 0.800 | 0.786 | 0.781 | 0.791 | 0.793 | **0.790** |
| Macro F1 | 0.800 | 0.786 | 0.781 | 0.791 | 0.793 | 0.790 |

Each run is a full leave-one-subject-out evaluation on 1152 held-out trials.
The seed-to-seed standard deviation is about 0.007, so **the figure to quote
is the mean, 0.790**, not the best seed.

### Per held-out subject (seed 12345)

| Held-out subject | Seed 12345 | Mean of five seeds | CSP + LDA with alignment |
| --- | ---: | ---: | ---: |
| S1 | 0.833 | 0.810 | 0.78 |
| S2 | 0.715 | 0.708 | 0.65 |
| S3 | 0.917 | 0.907 | 0.90 |
| S5 | 0.688 | 0.686 | 0.67 |
| S6 | 0.771 | 0.744 | 0.67 |
| S7 | 0.660 | 0.664 | 0.70 |
| S8 | 0.896 | 0.900 | 0.92 |
| S9 | 0.924 | 0.903 | 0.88 |
| **Mean** | **0.800** | **0.790** | **0.77** |

Seed 12345: 922 of 1152 held-out trials correct; predictions split 572 left and 580 right.

The CSP + LDA column is a separate classical analysis on the same aligned
2-4 second window and the same training trials. The model is ahead on the mean
by about 2 points; it is behind on subjects 7 and 8.

The five result files are `Version23_LOSO_seed*.json` in
`databci/bciupperlimbactionexperiment/ourexpv23/outputs/`. Version 22 gave
0.801, 0.787, 0.785, 0.789 and 0.791 for the same seeds (mean 0.791); the
differences are one or two trials per subject, the GPU variation described in
section 7.

### Within-subject experiment

The same data rules and model, trained and tested on one subject at a time
([train/within.py](train/within.py)).

| Setting | Value |
| --- | --- |
| Split | Stratified 10-fold inside each subject, shuffled with seed 42; every trial is tested exactly once |
| Training trials per fold | 129 or 130 (five batches per epoch) |
| Epochs | 60, fixed in advance |
| Logit centring | Off: a test fold has 14 or 15 trials of a subject the model was trained on |
| Everything else | As in LOSO: batch 32, Adam 0.001, weight decay 0.001, three models per fold |
| Evaluation | Once per fold, after the final epoch; nothing is selected on test trials |

The 60 epochs were chosen on validation splits taken from training folds only:
validation accuracy was 0.70 at 10 epochs, 0.74 at 20, and between 0.77 and
0.78 from 40 to 160 epochs.

| Subject | Within-subject accuracy | Range over the 10 folds | CSP + LDA within subject | LOSO, mean of five seeds |
| --- | ---: | --- | ---: | ---: |
| S1 | 0.826 | 0.67 to 0.93 | 0.798 | 0.810 |
| S2 | 0.764 | 0.53 to 0.87 | 0.629 | 0.708 |
| S3 | 0.889 | 0.79 to 1.00 | 0.916 | 0.907 |
| S5 | 0.646 | 0.40 to 0.86 | 0.535 | 0.686 |
| S6 | 0.660 | 0.43 to 0.86 | 0.739 | 0.744 |
| S7 | 0.701 | 0.50 to 0.93 | 0.674 | 0.664 |
| S8 | 0.965 | 0.93 to 1.00 | 0.966 | 0.900 |
| S9 | 0.951 | 0.79 to 1.00 | 0.891 | 0.903 |
| **Mean** | **0.800** | | **0.769** | **0.790** |

922 of 1152 trials are correct (macro F1 0.799); one run, seed 12345. File:
`Version23_Within_seed12345.json` in the same outputs folder.

- **Within-subject and LOSO give about the same mean** (0.80 and 0.79). With
  only 130 training trials from the subject, knowing the subject is worth
  about as much as having seven other subjects' 1008 trials.
- **It varies by subject.** Subjects 2, 7, 8 and 9 do better with their own
  data; subjects 5 and 6 do better when trained on other people.
- **The model is ahead of CSP + LDA within subject on the mean** (0.800
  against 0.769), mostly through subjects 2 and 5; it is behind on subjects 3
  and 6.
- **Fold scores are noisy.** A fold has 14 or 15 trials, so single folds range
  widely; only the per-subject total over 144 trials is meaningful.
- **Not comparable with the 74-76% of versions 10 to 16**, which reported the
  best test epoch of each fold.
- Per-subject alignment uses all trials of the subject, including each test
  fold's, without their labels.

The CSP + LDA column is a separate classical analysis (10-fold, aligned 2-4
second window, a different fold split).

### Which path matters (version 21, one seed each)

| Paths enabled | LOSO accuracy |
| --- | ---: |
| Graph head only | 0.721 |
| Wide path only | 0.742 |
| Graph head + wide path | 0.750 |
| Temporal encoder only | 0.785 |
| Wide path + temporal encoder | 0.785 |
| Graph head + temporal encoder | 0.793 |
| All three (this version) | 0.801 |

The temporal encoder carries most of the result. The steps above 0.785 are
about the size of the seed variation and come from single runs, so whether the
graph head and wide path add to the temporal encoder is **not established**.
All three are kept because the combination is the configuration that was
measured over five seeds.

## 5. What works, and the evidence

Each ingredient of this version, with what happened without it.

| Ingredient | Evidence | Measured on |
| --- | --- | --- |
| Analyse seconds 2-4 only | CSP + LDA within subject: 0.79 on seconds 2-4, 0.67 on the whole trial, 0.52 on seconds 0-2 (before the cue) | CSP analysis, v18 |
| Per-subject Euclidean alignment | CSP LOSO 0.67 without, 0.77 with. Graph model validation 0.65 without, about 0.69 with | v18, v19 |
| Log of the band powers | Raw band powers differ up to 15 times between subjects; without normalisation the model stayed at 0.50 even on its training subjects | v18 |
| Edge features on every electrode pair | With a graph that only records which pairs pass a threshold, LOSO was 0.50. Removing the edge values costs about 5 points in validation | v10-18 |
| Covariance next to PLV on the edges | About +0.01 in validation, within noise; kept because it is the natural input after alignment | v19 |
| Six bands up to 32 Hz | The 8-30 Hz filter leaves nothing above; the two higher bands of earlier versions were empty | v18 |
| Temporal encoder on the signals | LOSO 0.750 without it (graph head + wide path), 0.785 with it alone, 0.801 with all three | v20, v21 |
| Spatial filters in the temporal encoder | Per-electrode temporal features without spatial filters gave 0.75 to 0.79 in validation and were unstable; with spatial filters 0.78 at every checkpoint | v20 |
| Temporal dropout 0.7 | Validation 0.78 / 0.78 / 0.78 against 0.76 / 0.77 / 0.77 with 0.5 | v20 |
| Wide path | Validation 0.70-0.73 with it, 0.68-0.70 without, before the temporal encoder existed | v19 |
| Original trials only | With the temporal encoder, adding synthetic trials lowers LOSO from 0.802 to 0.767 | v20, v21 |
| Short schedule: 20 epochs, batch 32 | Validation accuracy drifts down after 10 to 20 epochs; the earlier 100-epoch, batch-8 schedule overfits the training subjects | v18-v20 |
| Shuffled batches and weight decay | The recording order groups trials by subject; unshuffled batches were part of the 0.50 results | v18 |
| Logit centring per held-out subject | +0.01 to +0.03 in validation and a balanced split between the classes (the uncentred model predicted one class for up to 65% of trials) | v19 |
| Three models per fold | Steadier accuracy from run to run; no gain in mean accuracy once logits are centred | v19 |

"Validation" means accuracy on training subjects held out one at a time, with
noise of about 0.02; it was used to choose settings without touching the test
subjects.

## 6. What does not work

None of the following is in this code base. Numbers are LOSO accuracy unless
they say validation. The baseline for comparison is 0.79 LOSO and about 0.78
validation.

### Data and graph construction

| Method | Result | Why (measured, or likely) |
| --- | --- | --- |
| Whole 4-second trial | CSP 0.67 within subject against 0.79 for seconds 2-4 | Measured: the first two seconds are before the cue |
| Later or longer window (2.5-6 s, 4-7 s) | CSP LOSO 0.71 and 0.52 against 0.77 | The response is strongest right after the cue |
| PLV graph thresholded at 0.35 with no edge values | 0.50 (versions 10 to 17) | The values on the edges carry the signal |
| Raw band powers without normalisation | 0.50; training accuracy also 0.50 | Measured: 15-fold scale differences between subjects |
| Per-trial standardisation on top of alignment | Validation 0.69-0.72 against 0.72-0.75 | Alignment already normalises; the extra step removes information |
| Separate mu and beta covariance on the edges | Validation 0.71 against 0.72 | No extra information for this model |
| Band powers per time segment | Validation 0.68 / 0.69, no gain | Trial-averaged power already captures it |

### Model

| Method | Result | Why |
| --- | --- | --- |
| GATv2 (three layers) | 0.50 on thresholded graphs; validation 0.48-0.50 on the graphs of this version | It cannot read edge features |
| Graph head without electrode embedding | Validation 0.46-0.49 | Electrode identity matters |
| Layer norm in place of batch norm | Validation 0.53-0.58 | Underfits |
| One or three layers, hidden size 64, stronger dropout | Within 0.01-0.02 of the default | No effect beyond noise |
| Electrode-aware readout (learned weights over electrodes and pairs) | Validation 0.75 / 0.72 / 0.71 against 0.77 / 0.78 / 0.78; training accuracy 0.80 to 0.99 | Measured overfitting to the training subjects |
| Temporal features added to node embeddings | Validation 0.75-0.79, unstable | No spatial filtering across electrodes |
| Temporal features routed into the attention (learned-band covariance in the pair stream) | Validation 0.73-0.75 | The attention does not exploit them |
| Other temporal encoder sizes (16 filters, 51-sample kernel, longer pooling) | Validation 0.75-0.77 | No effect beyond noise |
| Cropped training (random 1 to 1.75 second crops) | Validation 0.75-0.79, same as without | The window is already short |

### Training method

| Method | Result | Why |
| --- | --- | --- |
| Long schedule (100 epochs, batch 8) | Overfits; validation falls after 10-20 epochs | Too many updates on seven subjects |
| Adversarial subject classifier (gradient reversal) | Validation 0.78 at weight 0.1, down to 0.72 at weight 1.0 | Measured: removing subject identity from the features removes class information with it |
| Cross-subject supervised contrastive loss | 0.789 and 0.793 over three seeds against 0.791 | It does not reduce subject identity in the features |
| Proxy-Anchor metric learning | Validation 0.75 and 0.74 against 0.79 | With two classes the proxies duplicate the classifier |
| Test-time adaptation to the new subject: BatchNorm statistics, information maximisation, self-training, prototypes | Validation 0.784, 0.775, 0.781, 0.766 against 0.786 | Alignment and logit centring already use what the unlabeled trials offer |
| Choosing the best test epoch | About 75% within subject for any method (versions 10 to 17) | Selection on 14-trial test folds inflates the score |

### Synthetic data

| Method | Result | Why |
| --- | --- | --- |
| Diffusion generators trained 100 epochs, added to training | 0.50-0.51 with or without (GAT); validation unchanged for the graph model; 5 points lower with the temporal encoder | Measured: synthetic trials carry little of the real class signal (train on real, test on synthetic: 0.59 against 0.77) and are told from real trials with 0.99 accuracy |
| Lower guidance or EMA weights when sampling | No improvement; EMA output unusable | The defects come from the trained weights |
| Generators retrained for 1000 epochs, guidance 1.5 | 0.767 with the synthetic trials against 0.802 without | Measured: only two of eight generators reproduce the class signal; the subjects with weak real signal get useless generators |

## 7. How to read and report the numbers

- **Unlabeled test-subject data is used.** Alignment and logit centring both
  use the held-out subject's own trials without their labels. This is realistic
  (a short unlabeled recording from a new user) but must be stated. The result
  JSON records it under `unlabeled_test_subject_use`.
- **Quote the five-seed mean.** Single runs of the same model range from
  0.781 to 0.800. Differences below about 0.01 between methods are not findings.
- **GPU runs vary by a trial or two.** Two runs with the same seed can differ
  by one trial on a subject, because some GPU operations are not deterministic.
  On CPU the results are exactly reproducible.
- **How settings were chosen.** Model and training settings were chosen on
  training subjects held out one at a time. The 2-4 second window and the
  feature types came from classical analyses that used all eight subjects, so
  the result is mildly optimistic in that respect.
- **Older within-subject numbers are not comparable.** The 74-76% reported for
  versions 10 to 16 select the best test epoch on 14-trial folds; the same
  pipelines score 0.50 here.
- **Two subjects stay hard.** Subjects 5 and 7 are near 0.67 for this model
  and for CSP.

## 8. Ideas not tested

| Idea | Why it might help |
| --- | --- |
| Pre-train on another left/right motor-imagery dataset (for example Liu2024, already on disk) | The limit is seven training subjects; this adds subjects without changing the test protocol |
| Average the model's predictions with CSP + LDA | They make different mistakes on some subjects |
| Filter-bank or wider-band input for the temporal encoder | It currently sees only 8-30 Hz as one band |
| Weight training subjects by similarity to the new subject | Not all training subjects are equally useful |
| A short labelled calibration from the new subject | Changes the claim, but directly addresses the subject gap |
| Repeat the path ablation over several seeds | Would settle whether the graph head and wide path add to the temporal encoder |

## 9. Where this version comes from

| Version | LOSO accuracy | What changed |
| --- | ---: | --- |
| 10 to 17 | 0.50 | GATv2 on thresholded PLV graphs of the whole trial |
| 18 | 0.63 | Custom graph model, seconds 2-4, edge values, log inputs |
| 19 | 0.73 | Per-subject alignment, covariance edges, wide path, logit centring, ensemble |
| 20 | 0.79 | Temporal encoder; original trials only |
| 21 | 0.79 | Path ablation, synthetic-data diagnosis and retraining (no gain) |
| 22 | 0.79 | Five-seed baseline; invariance losses, proxy learning, test-time adaptation (no gain) |
| 23 | 0.79 | This version: the same method with only the code it needs |

Versions 18 to 22 are single-seed figures except where a mean is stated; the
0.801 and 0.802 quoted earlier for versions 20 and 21 are the best of five
seeds.

Version 23 was checked against version 22 before the old options were
dropped: it builds identical graphs from the raw recordings, starts from
identical weights, and after training steps on CPU its weights and outputs are
bit-identical. Detailed records of every experiment, with file names, are in
`src/ourexpv22/READMEFORMETHOD.md` and the result folders
`databci/bciupperlimbactionexperiment/ourexpv18` to `ourexpv23`.
