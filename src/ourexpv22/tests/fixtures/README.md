`legacy_expected.npz` contains outputs captured from the pre-refactor version ten
`dataset.py` and the preparation portion of `train.py:run_subject`, before any
implementation changes. The input MATLAB file is regenerated deterministically
by `tests/fixture_data.py` using NumPy default_rng seed 2026.

Keys `zero_x`, `zero_edges`, `zero_y` use threshold 0; `positive_x`,
`positive_edges`, `positive_y` use threshold 0.1. Each contains 12 trial graphs
with 22 channels and 8 feature bands in the original interleaved order.
Features are checked with rtol=1e-6, atol=1e-9; labels and edges match exactly.
Do not regenerate expected outputs from the new pipeline to resolve a mismatch.


`legacy_training.json` records the original monolithic trainer on the same
synthetic MATLAB fixture, after the dataset threshold was changed to 0.35 and
before training was split into modules. The capture used CPU, one Torch thread,
two folds, and `EPOCHS=4` (three iterations), with all other original settings.
It records exact fold indices, per-epoch train/test accuracies, and the final
mean/max/min. These are regression expectations, not scientific benchmark scores.
