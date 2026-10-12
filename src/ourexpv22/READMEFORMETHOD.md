# Method record: what works, what does not

This file tracks which methods have been tried for left/right motor imagery on
BCI Competition IV 2a (versions 10 to 22), what each one scored, and which did
not help. Update it when a new method is tested. For how to run the pipeline,
see [README.md](README.md).

Last updated: 2026-10-05 (version 22).

## How to read the numbers

Two kinds of numbers appear below. They are not interchangeable.

| Name | What it is | Noise |
| --- | --- | --- |
| **LOSO** | Leave one subject out, eight folds, evaluated once at the final epoch on 1152 held-out trials. This is the result to report | About ±0.012 to ±0.015 |
| **Validation** | One training subject held out at a time (four subjects, 576 trials), used to choose settings without touching the LOSO test subjects | About ±0.02 |

Validation numbers are given at epochs 10 / 20 / 30 where available. Almost
every number is a single run with one seed, so differences smaller than the
noise are not findings.

All LOSO results from version 19 on use the held-out subject's unlabeled
trials for alignment and logit centring. State this when reporting.

## Best method so far

**Version 20/21/22 default model, trained on original trials only: 0.791 LOSO
on average over five seeds** (0.785 to 0.802, standard deviation 0.007). The
0.801 and 0.802 quoted for versions 20 and 21 are the best of those five seeds
(seed 12345), not the typical value. A new method has to beat 0.791.

| Seed | 12345 | 100 | 200 | 300 | 400 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LOSO accuracy | 0.802 | 0.787 | 0.785 | 0.789 | 0.791 | **0.791** |

| Part | Setting |
| --- | --- |
| Trial window | Seconds 2 to 4 of each trial (after the cue), band-passed 8 to 30 Hz |
| Alignment | Per-subject Euclidean alignment, label-free |
| Graph | 22 electrodes; node features are six band powers; every electrode pair is an edge with PLV and covariance |
| Model | Sum of three paths: temporal encoder on the electrode signals (learned temporal and spatial filters, log power), GPS + GRIT graph head, and a linear "wide" path |
| Training | 20 epochs, batch 32, Adam 0.001, weight decay 0.001, shuffled batches, three seeds per fold averaged |
| Evaluation | Final epoch, with logit centring per held-out subject |
| Training data | 1008 original trials per fold; synthetic trials are not used |

## Results that can be compared (LOSO)

| Method | Version | LOSO accuracy |
| --- | --- | ---: |
| Temporal encoder + graph head + wide path, originals only | 20, 21 | **0.801** |
| Graph head + temporal encoder | 21 ablation | 0.793 |
| Temporal encoder only | 21 ablation | 0.785 |
| Wide path + temporal encoder | 21 ablation | 0.785 |
| CSP + LDA with alignment (classical reference, separate analysis) | | 0.77 |
| Graph head + wide path, originals only | 21 ablation | 0.750 |
| Wide path only | 21 ablation | 0.742 |
| Graph head + wide path with alignment, originals and synthetic | 19 | 0.727 |
| Graph head only | 21 ablation | 0.721 |
| Custom graph model, post-cue window, no alignment | 18 | 0.628 |
| GAT on thresholded PLV graphs, with or without synthetic trials | 10 to 17 | 0.499 to 0.512 |
| Custom graph model on the same thresholded graphs | 18, first run | 0.498 |

Reading:

- Every model with the temporal encoder beats every model without it.
- The top three are within noise of each other. The ablation does not
  establish that the graph paths add to the temporal encoder: the steps are
  10 and 9 trials out of 1152.
- Without the temporal encoder, the graph model (0.72 to 0.75) stays below the
  classical CSP reference (0.77).

Per held-out subject:

| Subject | v18 | v19 | v20/21 | CSP + LDA |
| --- | ---: | ---: | ---: | ---: |
| S1 | 0.653 | 0.681 | 0.833 | 0.78 |
| S2 | 0.563 | 0.639 | 0.715 | 0.65 |
| S3 | 0.583 | 0.896 | 0.917 | 0.90 |
| S5 | 0.500 | 0.625 | 0.681 | 0.67 |
| S6 | 0.632 | 0.681 | 0.778 | 0.67 |
| S7 | 0.507 | 0.542 | 0.667 | 0.70 |
| S8 | 0.785 | 0.868 | 0.896 | 0.92 |
| S9 | 0.806 | 0.889 | 0.924 | 0.88 |
| **Mean** | 0.628 | 0.727 | 0.801 | 0.77 |

## The older within-subject results (not comparable)

The root `Readme.md` lists within-subject 10-fold accuracy for versions 10, 14,
15 and 16:

| | Version 10 | Version 14 | Version 15 | Version 16 |
| --- | ---: | ---: | ---: | ---: |
| Mean accuracy | 74.44% | 75.98% | 74.91% | 75.48% |

These cannot be ranked against the LOSO numbers, and should not be used to
rank the versions against each other:

- The reported score is the **best test-fold epoch** out of 100 to 249 epochs.
- Each test fold has 14 or 15 trials. Choosing the best of 100 or more epochs
  on 14 trials gives roughly 75% even for a classifier at chance.
- The same pipelines score 0.50 on LOSO, and the table is nearly flat across
  subjects (72 to 80%), while the real signal differs widely between subjects
  (CSP within-subject: 0.53 for S5, 0.97 for S8).

The inflation is an interpretation; those versions have not been rerun with
final-epoch evaluation. Version 20/21 has never been run within-subject.

## Methods that did not give good results

Each entry says what was tried, what it scored, the likely reason, and whether
the option is still in the code. "Likely reason" is an interpretation unless
it says measured.

### Data and graph construction

| Method | Tried in | Result | Likely reason | In the code |
| --- | --- | --- | --- | --- |
| Whole 4-second trial (seconds 0 to 4) | 10 to 17 | CSP + LDA: 0.67 within subject, against 0.79 for seconds 2 to 4. Seconds 0 to 2 alone: 0.52 | Measured: the first two seconds are before the cue and carry no class signal | `legacy_dataset_config()` |
| Later or longer window | 18 analysis | CSP LOSO with alignment: 0.711 for seconds 2.5 to 6 and 0.518 for 4 to 7, against 0.770 for 2 to 4 | The response is strongest right after the cue; also needs re-extraction, since the bundle stores seconds 0 to 4 | Not implemented |
| PLV graph thresholded at 0.35, no edge values | 10 to 17 | LOSO 0.50. On the newer graphs, removing edge weights costs about 5 points in validation (0.55 to 0.58 against 0.60 to 0.65) | The values on the edges carry the signal, not which edges pass a threshold | `graphs.mode="legacy"` or `"threshold"` |
| Raw band powers as node features, no normalisation | 18, first run | LOSO 0.498; training accuracy stayed at 0.50 | Measured: band power differs up to 15 times between subjects | `input_norm="none"` |
| Eight bands, 8 to 40 Hz | 10 to 17 | Two bands (32 to 40 Hz) are about zero | The 8 to 30 Hz filter is applied first | `legacy_dataset_config()` |
| No per-subject alignment | 18 | Validation 0.65, against about 0.69 with alignment; CSP LOSO 0.67 against 0.77 | Subjects differ in scale and channel mixing | `preprocessing.alignment="none"` |
| Separate mu and beta covariance on the edges | 19 | Validation 0.71 / 0.70, against 0.72 / 0.69 for one 8 to 30 Hz band | No extra information for this model | `connectivity.covariance_bands` |
| Band powers per time segment (four segments) | 19 | Validation 0.68 / 0.69, no gain | Trial-averaged power already captures it | `features.segments` |
| Per-trial standardisation of inputs together with alignment | 19 | Validation 0.69 to 0.72, against 0.72 to 0.75 with plain log inputs | Alignment already normalises; the extra step removes information | `input_norm="log_zscore"`, `edge_norm="zscore"` |

### Model: graph side

| Method | Tried in | Result | Likely reason | In the code |
| --- | --- | --- | --- | --- |
| GAT (GATv2, three layers) | 10 to 17; 18 screening | LOSO 0.50 on the old graphs; validation 0.48 to 0.50 on the new graphs | It cannot read edge features, and the complete graph has no informative topology | `--model gat` |
| Model-size and regularisation changes on the old graphs (layer norm, hidden 64, two layers, dropout 0.5, head dropout, local-only, global-only, no random-walk encoding) | 18 | Validation 0.46 to 0.53 for all of them | The features were the problem, not the model | Various options |
| No electrode embedding | 18 | Validation 0.46 to 0.49 | Electrode identity matters with fixed electrodes | `num_electrodes=None` |
| Layer norm in place of batch norm | 18 | Validation 0.53 to 0.58, training accuracy only 0.54 | Underfits in this setup | `layer_norm=True` |
| One layer, three layers, dropout 0.5, wide-path dropout 0.8 | 19 | Validation within 0.01 to 0.02 of the default | No effect beyond noise | Options |
| Graph head and wide path without the temporal encoder | 19, 21 | LOSO 0.727 (with synthetic) and 0.750 (originals only) | Trial-averaged features cap it at about the level of a linear classifier (0.73 to 0.74) | `--paths graph wide` |
| Electrode-aware readout (learned signed weights over electrodes and pairs in place of mean pooling) | 21 | Validation: graph only 0.72 / 0.69 / 0.71 (mean pooling 0.72 / 0.72 / 0.71); default model 0.75 / 0.72 / 0.71 (mean pooling 0.77 / 0.78 / 0.78). Training accuracy rises from 0.80 to 0.99 | Measured overfitting: more freedom in the graph head fits the training subjects and does not transfer | `readout="electrode"` |

### Model: temporal features

| Method | Tried in | Result | Likely reason | In the code |
| --- | --- | --- | --- | --- |
| Temporal `node` path: per-electrode log power added to node embeddings | 20 | Validation 0.75 / 0.73 / 0.74 with synthetic; 0.75 / 0.79 / 0.76 originals only, unstable | No spatial filtering across electrodes | `temporal="node"` |
| Temporal `both` (node and head paths) | 20 | Validation 0.79 / 0.77 / 0.76, drifting down; no better than `head` | The node path adds nothing to the head path | `temporal="both"` |
| Temporal `edge` path: learned-band covariance into the attention's pair stream | 21 | Validation 0.75 / 0.74 / 0.73; with `head` 0.77 / 0.77 / 0.76 | The attention does not exploit it; same level as the graph model without it | `temporal="edge"`, `"edge+head"` |
| Other temporal encoder sizes: 16 filters, 51-sample kernel, 125-sample pooling, dropout 0.5 | 20 | Validation 0.75 to 0.77, same or lower than the default (0.78) | No effect beyond noise | `temporal_*` options |
| Temporal encoder without the wide path | 20 | Validation 0.75 / 0.74 / 0.75, against 0.76 / 0.77 / 0.77 | Small loss; in LOSO, graph + temporal reaches 0.793 | `--paths graph temporal` |
| Cropped training (random 1 to 1.75 second crops, averaged at test time) | 21 | Validation 0.75 to 0.79, same as no cropping (0.77 / 0.78 / 0.78) | The 2-second window is already short | `temporal_crop` |

### Training and evaluation

| Method | Tried in | Result | Likely reason | In the code |
| --- | --- | --- | --- | --- |
| Unshuffled training batches | 10 to 18 first run | Part of the 0.50 LOSO results; shuffling alone raised training accuracy only slightly | The saved order groups trials by subject | `train_shuffle=False` |
| Long schedules for LOSO (100 epochs, batch 8) | 18 first run | Used for the 0.498 run; in later screening, validation accuracy drifts down after 10 to 20 epochs at batch 32 | Overfits the training subjects | `--epochs`, `--batch-size` |
| Best-test-epoch selection | 10 to 17 within-subject | Gives about 75% regardless of method | Selection on 14-trial test folds | `train/train.py`, `train/train_syn.py` |
| Seed ensemble as an accuracy gain | 19 | Raw validation 0.729 against 0.701; no difference once logits are centred (0.720) | It stabilises, it does not add signal | `--ensemble` (kept on for stability) |

### Training method

| Method | Tried in | Result | Likely reason | In the code |
| --- | --- | --- | --- | --- |
| Adversarial subject classifier with gradient reversal | 22 | Validation 0.778 at weight 0.1, down to 0.72 at weight 1.0 (baseline 0.78) | Measured: subject identity and class information are entangled in the features; removing one removes the other | `--invariance adversarial` |
| Cross-subject supervised contrastive loss | 22 | LOSO 0.789 and 0.793 over three seeds, against 0.791 for the baseline | It does not reduce subject identity in the features (probe 0.77) | `--invariance contrastive` |
| Proxy-Anchor metric learning loss | 22 | Validation 0.749 and 0.737 at weights 0.1 and 0.3, against 0.786 | With two classes the proxies duplicate the classifier; the extra loss overfits the training subjects | `--invariance proxy` |
| Test-time adaptation to the held-out subject: BatchNorm statistics, information maximisation, self-training, prototypes | 22 | Validation 0.784, 0.775, 0.781, 0.766, against 0.786 without | Alignment and logit centring already use what the unlabeled trials offer | `train/adaptation.py` |
| Judging methods from a single seed | 20, 21 | The same model gives 0.785 to 0.802 across five seeds | Run-to-run variation of about 0.007 | `--seed` |

### Synthetic data

| Method | Tried in | Result | Likely reason | In the code |
| --- | --- | --- | --- | --- |
| Adding the current synthetic trials to training | 14 to 19 | LOSO: 0.499 to 0.512 with, 0.511 without (GAT). Validation in 19: no difference. CSP LOSO: 0.766 with, 0.760 without | Measured: the synthetic trials carry little of the real class signal (train on real, test on synthetic: 0.59, against 0.77 on real) | `--synthetic` |
| Synthetic trials with the temporal encoder | 20 | Validation 0.70 to 0.75 with, 0.76 to 0.77 without; training accuracy 1.00 | Measured: synthetic trials are separable by class before the cue (0.72, real 0.51) and are told from real with 0.99 accuracy; the encoder learns that pattern | `--synthetic` (off by default) |
| Lower guidance scale when sampling (1.0, 1.5, 2.5 against 4.0) | 21 | Removes the pre-cue class difference (0.69 to 0.52) and the class signal with it (real to synthetic 0.64 to 0.56). Spectrum unchanged (13 to 30 Hz power 2.1 times real) | Guidance was creating the class difference; the spectrum comes from the weights | `datasetsynthetic/resample.py` |
| Generators retrained for 1000 epochs, guidance 1.5 (bundle `datasynv1`) | 21 | LOSO 0.767 with the synthetic trials, 0.802 without | Measured: only two of eight generators reproduce the class signal; all remain unrealistic | `--bundle .../datasynv1/fold_cross --synthetic` |
| Sampling from EMA weights | 21 | Output has tens of thousands of times the real power | After 100 epochs the EMA still holds early weights | `resample.py --weights ema` |
| Generators trained for 100 epochs | 14 to 21 bundles | Six of eight had their best validation loss at epoch 99. Pilot: no class signal at 100 epochs, 0.83 to 0.88 at 1000 | Measured: under-trained | `generate --epochs` |
| Rescaling the generator input to unit variance, at 100 epochs | 21 pilot | real to synthetic 0.55, against 0.53 for the bundle's scale | Training length, not scale, was the limit | Not in the code |
| Generator batch size 8 at 1000 epochs | 21 pilot | real to synthetic 0.73 only at guidance 4.0, against 0.87 at guidance 1.5 for batch size 1 | Eight times fewer weight updates | Not in the code |

## Generator pilot (version 21, subject 8 only)

A local pilot with the same generator model and optimiser, scored with
`datasetsynthetic/evaluate.py` measures on 60 synthetic trials (250 sampling
steps). Real to real for this subject is 0.966. One subject, so accuracies
carry several points of noise. Files:
`ourexpv21/generator_pilot_20261004T104317Z/`.

| Generator input | Epochs, batch | Guidance | real->syn | syn->real | syn->syn before the cue | real vs syn |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Raw / 100 (the bundle's setup) | 100, 1 | 4.0 | 0.53 | 0.54 | 0.63 | 0.98 |
| Raw / own standard deviation | 100, 1 | 4.0 | 0.55 | 0.65 | 0.62 | 0.985 |
| Aligned, 8-30 Hz, seconds 2-4 | 100, 1 | 4.0 | 0.58 | 0.47 | | 1.00 |
| Aligned, 8-30 Hz, seconds 2-4 | 1000, 8 | 4.0 | 0.73 | 0.85 | | 1.00 |
| Aligned, 8-30 Hz, seconds 2-4 | 1000, 1 | 1.5 | 0.87 | 0.88 | | 1.00 |
| Aligned, 8-30 Hz, seconds 2-4 | 1000, 1 | 4.0 | 1.00 | 0.86 | | 1.00 |
| Raw / 100 (the bundle's setup) | 1000, 1 | 1.5 | 0.83 | 0.83 | 0.72 | 0.985 |
| Raw / 100 (the bundle's setup) | 1000, 1 | 4.0 | 1.00 | 0.91 | 1.00 | 0.985 |

What it shows:

- **Training length was the main problem.** At 1000 epochs both inputs give
  synthetic trials that carry the real class signal (0.83 to 0.88 in both
  directions at guidance 1.5); at 100 epochs neither does.
- **Rescaling the input did not help by itself** at 100 epochs. The small input
  scale (standard deviation about 0.1) is real, but it was not the cause.
- **Guidance 4.0 exaggerates the classes** (1.00); 1.5 is the better setting.
- **Raw-input generators still separate the classes before the cue** (0.72),
  where real trials do not. The classifier does not use that part.
- **Realism is still poor for every setting.** Real and synthetic trials are
  told apart with 0.98 to 1.00 accuracy. The raw-input generator has 1.6 times
  the real power at 13-30 Hz; the window-input generator puts about half of
  its power outside 8-30 Hz. Neither changed between epochs 250 and 1000, nor
  with 250, 1000 or DDIM-100 sampling steps.
- **No memorisation:** synthetic trials are no closer to their nearest
  training trial than real trials are to each other.
- **Batch size 8 is four times cheaper and clearly weaker** at the same number
  of epochs.
- **Time on the GTX 1060:** 86 minutes per subject for the window input and
  112 minutes for the raw input, at 1000 epochs and batch size 1.

Not yet known: whether these class-correct but unrealistic trials improve the
classifier. That needs generators for all eight subjects and a LOSO run with
and without them.

## Retrained generators, all eight subjects (bundle `datasynv1`)

Generators retrained for 1000 epochs at batch size 1, sampled from the final
epoch with guidance 1.5 (about 18 hours on the GTX 1060). Same original
trials and folds as the old bundle.

Synthetic-trial check (`evaluate.py`, CSP + LDA, seconds 2-4):

| Subject | real->real | real->syn | syn->real | real vs syn |
| --- | ---: | ---: | ---: | ---: |
| S1 | 0.798 | 0.520 | 0.694 | 1.000 |
| S2 | 0.629 | 0.630 | 0.535 | 0.979 |
| S3 | 0.916 | 0.640 | 0.785 | 0.938 |
| S5 | 0.535 | 0.560 | 0.465 | 0.996 |
| S6 | 0.739 | 0.580 | 0.549 | 0.996 |
| S7 | 0.674 | 0.620 | 0.562 | 1.000 |
| S8 | 0.966 | 0.830 | 0.910 | 0.988 |
| S9 | 0.891 | 0.820 | 0.903 | 0.996 |
| **Mean** | 0.769 | 0.650 | 0.675 | 0.987 |
| Old bundle (100 epochs, guidance 4.0), mean | 0.769 | 0.587 | 0.569 | 0.993 |

Classifier result (LOSO, default model, same settings and seeds):

| Held-out subject | Originals only | Originals + new synthetic | Difference |
| --- | ---: | ---: | ---: |
| S1 | 0.833 | 0.764 | -0.069 |
| S2 | 0.722 | 0.667 | -0.056 |
| S3 | 0.917 | 0.903 | -0.014 |
| S5 | 0.688 | 0.618 | -0.069 |
| S6 | 0.771 | 0.736 | -0.035 |
| S7 | 0.667 | 0.667 | 0.000 |
| S8 | 0.896 | 0.882 | -0.014 |
| S9 | 0.924 | 0.903 | -0.021 |
| **Mean** | **0.802** | **0.767** | **-0.035** |

What it shows:

- **The retrained synthetic trials still lower accuracy**, by 3.5 points (40
  trials of 1152), on seven of eight subjects and never higher.
- **The generators are better but only good for two subjects.** Subjects 8
  and 9 reach 0.82 to 0.91; subjects 2, 5, 6 and 7 are near chance, so their
  synthetic trials are labelled trials without the class signal.
- **A generator is only as good as its subject's real signal.** The subjects
  whose generators fail are those whose own real trials are hard to classify
  (real to real 0.54 to 0.74).
- **Realism did not improve** (0.987 against 0.993), and the classes still
  differ before the cue (0.673).

Files: `ourexpv21/outputs/Version21_SyntheticEvaluation_datasynv1.json` and
`Version21_GritGPS_LOSO_datasynv1_*_20261005T111223Z.json`.

## Subject-invariant training (version 22)

Two extra training losses on the model's learned features
(`train/invariance.py`, `--invariance`), meant to stop the model relying on
subject-specific patterns.

A probe shows the features do carry subject identity: a linear classifier
tells the six training subjects apart with 0.73 accuracy (chance 0.17).

Validation on held-out training subjects (epochs 10 / 20 / 30, one seed):

| Loss | Weight | Validation accuracy | Subject probe |
| --- | ---: | --- | ---: |
| None | | 0.776 / 0.781 / 0.780 | 0.73 |
| Adversarial subject classifier (gradient reversal) | 0.1 | 0.778 / 0.767 / 0.778 | 0.67 |
| | 0.3 | 0.762 / 0.747 / 0.727 | 0.61 |
| | 1.0 | 0.717 / 0.715 / 0.727 | 0.56 |
| Cross-subject supervised contrastive | 0.1 | 0.788 / 0.793 / 0.788 | 0.77 |
| | 0.3 | 0.766 / 0.785 / 0.800 | 0.76 |
| | 1.0 | 0.776 / 0.762 / 0.774 | 0.77 |

Full LOSO for the contrastive loss, same seeds as the baseline:

| Method | Seed 12345 | Seed 100 | Seed 200 | Mean of the three |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.801 | 0.787 | 0.785 | 0.791 |
| Contrastive, weight 0.1 | 0.794 | 0.783 | 0.791 | 0.789 |
| Contrastive, weight 0.3 | 0.808 | 0.781 | 0.788 | 0.793 |

What it shows:

- **No gain.** The contrastive loss differs from the baseline by -0.001 and
  +0.002 on average, far inside the seed variation of 0.007.
- **The adversarial loss removes subject information and class information
  together**: the probe falls to 0.56 and validation accuracy falls by up to
  five points.
- **The contrastive loss does not make features subject-free** (probe
  unchanged), so it had no mechanism to help.

Files: `ourexpv22/outputs/Version22_GritGPS_LOSO_OriginalOnly_*_seed*.json`.

## Test-time adaptation and proxy learning (version 22)

Screened on four held-out training subjects, three seeds each, 20 epochs, one
model per run (576 trials times three seeds). The held-out subject's labels
are used only for scoring.

| Training loss | No adaptation | BatchNorm statistics | Information maximisation | Self-training | Prototypes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | **0.786** | 0.784 | 0.775 | 0.781 | 0.766 |
| Proxy-Anchor, weight 0.1 | 0.749 | 0.747 | 0.751 | 0.748 | 0.730 |
| Proxy-Anchor, weight 0.3 | 0.737 | 0.739 | 0.739 | 0.743 | 0.733 |

What it shows:

- **No test-time adaptation method helps.** Recomputing BatchNorm statistics
  and self-training are level with doing nothing; information maximisation
  and prototypes are 1 to 2 points lower. Per-subject alignment and logit
  centring, which are already in the pipeline, appear to capture what the
  unlabeled trials of a new subject can give.
- **Proxy-Anchor training lowers accuracy by 4 to 5 points.** With two
  classes the proxies duplicate the classifier, and the extra loss fits the
  training subjects harder.
- Neither was run on the full LOSO, because nothing passed screening.

Code: `train/adaptation.py` (not wired into the trainer) and
`--invariance proxy`.

## Not tested yet

| Idea | Why it might help | Cost |
| --- | --- | --- |
| Use synthetic trials only from generators that pass the check (subjects 8 and 9, perhaps 3), or keep only synthetic trials that a real-trained classifier labels correctly | Half of the current synthetic trials carry no class signal | About 10 minutes per LOSO run plus a small trainer option |
| Give synthetic trials a lower weight in the loss, or pre-train on them and fine-tune on real trials | Limits the damage of unrealistic trials | Small trainer change |
| Remove the broadband noise in generated trials (EMA sampling, other prediction target, longer training) | Realism is still poor after 1000 epochs | About 90 minutes per test on one subject |
| Train the generators on the aligned, band-passed 2 to 4 second window only | The generator would model only what the classifier uses | Code changes in the generator data, the bundle format and the loader |
| Repeat the ablation (temporal only, graph + temporal) with several seeds | The ablation differences of 1 to 2 points are within the seed variation now measured | About 8 minutes per run |
| Train the shared layers on all four classes (left, right, feet, tongue) | Doubles the real trials per subject | Needs the other two classes extracted |
| Augment with real signal (same-class mixing, segment recombination, time shifts) | Keeps the real class signal by construction | Under an hour per test |
| Within-subject evaluation of version 21 at the final epoch | Gives a fair number against the older within-subject table | Under an hour |
| Subject 4 | One more training and test subject; its file layout needs checking | Small |

## Where the evidence is

| Evidence | Location |
| --- | --- |
| LOSO result files and logs | `databci/bciupperlimbactionexperiment/ourexpv18` to `ourexpv22`, folder `outputs/` |
| Ablation runs | `ourexpv21/outputs/Version21_Ablation_*_20261004T051101Z.json` |
| Synthetic-data evaluation | `ourexpv21/outputs/Version21_SyntheticEvaluation_fold_cross.json` |
| Re-sampling sweep | `ourexpv21/resampling_20261004T075522Z/` |
| Generator pilot | `ourexpv21/generator_pilot_20261004T104317Z/` (script, log, samples, weights, result files) |
| Validation screening tables | History section of [README.md](README.md) |

The validation screening and the CSP analyses were run with scratch scripts
that are not in the repository; only their results are recorded.
