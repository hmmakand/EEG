"""Validation splits of a fold's training side, shared by the papers that train several models per fold."""
import numpy as np


def split_groups(groups, parts: int, seed: int, key: int):
    """Shuffle the distinct `groups` (seeded by seed and key) and divide them into `parts` lists."""
    unique = np.unique(groups)
    rng = np.random.default_rng([seed, key])
    return [sorted(part.tolist()) for part in np.array_split(rng.permutation(unique), parts)]
