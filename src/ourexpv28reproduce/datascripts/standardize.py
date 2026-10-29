"""braindecode's exponential moving standardisation, used by several papers' pre-processing."""
import numpy as np
import pandas as pd


def exponential_moving_standardize(data, factor: float = 1e-3, init_block: int | None = 1000, eps: float = 1e-4):
    """braindecode.preprocessing.exponential_moving_standardize on [time, channels]: subtract an exponential
    moving mean and divide by an exponential moving standard deviation (decay 1 - factor); the first
    `init_block` samples are standardised with their own mean and standard deviation."""
    frame = pd.DataFrame(data)
    demeaned = frame - frame.ewm(alpha=factor).mean()
    standardized = np.array(demeaned / np.maximum(eps, np.sqrt(np.array((demeaned * demeaned).ewm(alpha=factor).mean()))))
    if init_block:
        block = data[:init_block]
        standardized[:init_block] = (block - block.mean(axis=0, keepdims=True)) / np.maximum(
            eps, block.std(axis=0, keepdims=True))
    return standardized
