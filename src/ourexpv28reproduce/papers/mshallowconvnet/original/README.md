# Original code

Copied unchanged from https://github.com/comojin1994/m-shallowconvnet (branch
`main`, downloaded 2026-10-16), for reference only; nothing here is imported.
The PyTorch-Lightning code needs `pytorch_lightning`, `easydict` and an old
`braindecode`; the port in `../network.py` and `../training.py` follows it.

The repository has no licence file. These copies are kept for documentation
of the reproduction and are not to be redistributed.

Files: `model/` (the network and the max-norm layers), `dataloader/` (BCI IV
2a/2b loading, pre-processing and the CutCat augmentation),
`utils/training_utils.py` (optimiser, scheduler, checkpoints), `configs/`
(hyper-parameters), `training.py` and `evaluation.py`.
