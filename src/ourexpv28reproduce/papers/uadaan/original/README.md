# Reference code (not the authors')

The UA-DAAN paper (Shen et al. 2025) publishes no code. The implementation
followed here is https://github.com/chenjlbj/EEG-Depression-MSUA-DAAN
(branch `main`, downloaded 2026-10-17; created 2025-11), a third-party
implementation of UA-DAAN and its multi-source extension MSUA-DAAN. Its
UA-DAAN parts are copied here unchanged, for reference only; nothing here is
imported. The repository has no licence file; these copies document the
reproduction and are not to be redistributed.

- `bayesian.py`: Bayesian linear layer and its KL loss
- `extractor.py`: feature extractor (MLP with dropout)
- `classifier.py`: Bayesian label classifier
- `discriminator.py`: global and local domain discriminators, uncertainty-weighted entropy loss, gradient reversal, dynamic adversarial factor
- `model.py`: UA-DAAN (10 Monte Carlo samples)
- `main.py`, `ua-daan12fold.py`: training and evaluation scripts
- `dataloader.py`, `tenfold.py`: its data files and subject-independent folds

The DBConformer repository (https://github.com/wzwvv/DBConformer) contains no
UA-DAAN model; its `utils/network.py` has only generic adversarial parts
(gradient reversal, discriminators).
