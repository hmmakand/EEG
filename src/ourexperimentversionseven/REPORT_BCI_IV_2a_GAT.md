# Report: `kaggleeeg.ipynb`

## Goal
Binary motor-imagery classification (left hand vs. right hand) from EEG, using a **Graph Attention Network (GAT/GATv2)** where each EEG trial is represented as a graph over electrodes, evaluated per-subject on the BCI Competition IV Dataset 2a.

## 1. Dataset

**Source:** BCI Competition IV Dataset 2a (`A0{subject}T.mat` files), loaded from `data/kaggle/bci-competition-iv-data-sets-2a/` (or `/kaggle/input/...` when run on Kaggle).

**Per subject:**
- 22 EEG channels (EOG channels dropped), sampled at 250 Hz.
- Trials are 4 s long → 1000 samples/trial, extracted starting at each cue's `trial` onset index.
- Only labels `1` (left hand) and `2` (right hand) are kept — `3` (feet) and `4` (tongue) are discarded, making this a 2-class problem.
- Subjects processed: **1, 2, 3, 5, 6, 7, 8, 9** (subject 4 is skipped — not explained in the notebook, likely a known-bad/missing file for this dataset).
- Each subject ends up with 72 left + 72 right trials → **144 graphs per subject**.

**Note:** The notebook contains *two* separate loader implementations (`load_bci_competition_data` defined twice, cells `97e48757` and `565863bc`) — an early exploratory one using `struct_as_record=False`, and a second one using raw `.mat` struct indexing (`data[0, session_idx][0,0]`) that is what's actually used in the final pipeline (cell `2793b17a`).

## 2. Preprocessing

1. **Bandpass filter:** 5th-order Butterworth, 8–30 Hz (covers mu and beta rhythms, the classic motor-imagery bands), applied with zero-phase filtering (`sosfiltfilt`).
2. **Connectivity (edges):** Phase Locking Value (PLV) between every electrode pair, computed via the Hilbert transform's instantaneous phase. This produces a 22×22 PLV matrix per trial — used as the graph's adjacency/edge weights.
3. **Graph thresholding:** Edges are kept if `PLV > threshold`, with `threshold = 0` in the final run — i.e., effectively a **fully connected graph** per trial (all 22×21 possible directed edges included), so PLV values act as edge weights rather than as a sparsifying filter.
4. **Node features (band power):** For each electrode, power is computed in 8 frequency sub-bands spanning 8–40 Hz in 4 Hz steps (`[8,12,16,20,24,28,32,36,40]`), each first bandpass-filtered to that sub-band, then band power estimated via Welch's PSD + Simpson's-rule integration. Result: each node (electrode) gets an **8-dimensional feature vector** (one power value per sub-band).

So each trial → one graph with **22 nodes**, **8-dim node features**, and a dense weighted adjacency from PLV.

## 3. Model: GATv2

```
GAT(
  conv1: GATv2Conv(8 → 22, heads=3, concat=True)   # in: 8 freq-band features
  gn1:   GraphNorm(66)
  conv2: GATv2Conv(66 → 22, heads=3, concat=True)
  gn2:   GraphNorm(66)
  conv3: GATv2Conv(66 → 22, heads=3, concat=True)
  gn3:   GraphNorm(66)
  readout: global_mean_pool
  dropout: p=0.5
  lin:   Linear(66 → 2)
)
```

- 3 stacked GATv2 attention layers (`hidden_channels=22`, `heads=3`), each followed by `GraphNorm` for normalization across the graph batch.
- Note: in the **production model** used for training (cell `1c18b945`), edge weights (PLV) are *not* passed into `GATv2Conv` — only `edge_index` is used, so the model relies purely on graph structure (which edges exist) plus learned attention, not on the PLV magnitude itself. (This differs from the earlier exploratory model in cell `9516eac6`, which does pass `edge_attr`/PLV weight into the conv via `edge_dim=1`.)
- Global mean pooling produces a graph-level embedding, dropout (0.5) regularizes it, then a linear layer outputs 2 logits (left/right).

## 4. Training Strategy

- **Per subject, independently** (no cross-subject transfer — a separate model is trained from scratch for each of the 8 subjects).
- **10-fold cross-validation** (`KFold(n_splits=10, shuffle=True, random_state=42)`) over that subject's 144 graphs.
- For each fold: `batch_size=32`, `Adam` optimizer (`lr=0.001`), `CrossEntropyLoss`, trained for **249 epochs** (`range(1, 250)`).
- At every epoch, train and test accuracy are computed; the checkpoint with the **highest test accuracy** seen during that fold's training is kept (i.e., best-epoch selection on the test fold itself — see caveat below).
- The best test accuracy from each of the 10 folds is collected; **mean/max/min** across folds is reported per subject.

## 5. Results

| Subject | Mean | Max | Min |
|---|---|---|---|
| S1 | 0.7510 | 0.8571 | 0.6000 |
| S2 | 0.7438 | 0.8000 | 0.6667 |
| S3 | 0.7276 | 0.9333 | 0.5714 |
| S5 | 0.7576 | 0.8571 | 0.6667 |
| S6 | 0.6952 | 0.7857 | 0.5333 |
| S7 | 0.7152 | 0.8667 | 0.6000 |
| S8 | 0.7429 | 0.9333 | 0.5714 |
| S9 | 0.7710 | 0.8667 | 0.6000 |

Overall mean across subjects ≈ **0.735** (73.5%) accuracy for binary left/right motor-imagery classification, saved to `BCI_IV_2a_GAT_Results.json`. The final cells build box plots visualizing the spread of per-subject mean accuracies.

## Caveats worth knowing
- **No fixed validation set separate from test:** epoch/model selection ("optimal" checkpoint) is chosen by peeking at test-fold accuracy during training, which optimistically biases the reported numbers — there's no held-out set used purely for early stopping.
- **Fully connected graphs:** with `threshold=0`, the "graph" structure barely constrains the GAT — it's close to a fully-connected attention model over 22 nodes, so the graph-topology benefit here mainly comes from PLV-weighted attention initialization rather than sparsity.
- **Edge weights unused in the trained model:** the `GAT` class actually used in the sweep (cell `1c18b945`) ignores PLV edge weights in the conv layers.
- **Duplicate/leftover code:** the notebook has two loader functions and two model/pipeline implementations (an early single-subject demo, then the real multi-subject sweep) — only the second (`2793b17a` + `1c18b945` + `8bad463b`/`565863bc`/`57c9aca1`) produced the saved results.
