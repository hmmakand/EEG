# Original code

Copied unchanged from https://github.com/guoruilu/SlimSeiz (branch `master`,
downloaded 2026-10-17), for reference only; nothing here is imported. The
repository has no licence file; these copies document the reproduction and
are not to be redistributed.

- `Loop_select_ch_PCA_SMOTE_DT.ipynb`: the channel selection (per channel:
  PCA, SMOTE, decision tree; 30 repetitions) on CHB-MIT.
- `Loop_ADDNet_kFold.ipynb`: the network (convolutions + Mamba block, the
  `OneDCNN` class), its training (cross-entropy + supervised contrastive
  loss) and the 10-fold evaluation on CHB-MIT.
- `readme_original.md`: the repository's readme.

The SRH-LEI notebooks of the repository are not copied (same code, other dataset).
