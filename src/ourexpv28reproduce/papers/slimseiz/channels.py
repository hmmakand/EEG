"""SlimSeiz's channel selection (Section II-B, Loop_select_ch_PCA_SMOTE_DT.ipynb).

For every channel alone: split the segments 70/30 at random, reduce each
(down-sampled) segment to 60 principal components (fitted on the training
part), balance the training part with SMOTE, train a decision tree and keep
its test accuracy. Rank the channels by accuracy and keep the best k. Repeat
m = 30 times and select the k channels that are most often among the best k.

Here the selection sees only the training side of a fold (the training
subjects, or the training folds of a subject), never its test samples; the
code ran it on all of a patient's data.
"""
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from .settings import ChannelSelectionSettings


def smote(x, y, neighbours: int, rng):
    """SMOTE (Chawla et al. 2002), as imblearn's default: new minority samples on the lines to their k nearest
    minority neighbours, until the classes are equal."""
    classes, counts = np.unique(y, return_counts=True)
    new_x, new_y = [x], [y]
    for label, count in zip(classes, counts):
        missing = counts.max() - count
        own = x[y == label]
        if missing == 0 or len(own) < 2:
            continue
        k = min(neighbours, len(own) - 1)
        nearest = NearestNeighbors(n_neighbors=k + 1).fit(own).kneighbors(own, return_distance=False)[:, 1:]
        base = rng.integers(len(own), size=missing)
        partner = own[nearest[base, rng.integers(k, size=missing)]]
        new_x.append(own[base] + rng.random((missing, 1)) * (partner - own[base]))
        new_y.append(np.full(missing, label))
    return np.concatenate(new_x), np.concatenate(new_y)


def channel_accuracy(x, y, channel: int, settings: ChannelSelectionSettings, rng) -> float:
    segments = x[:, channel, ::settings.downsample]
    train_x, test_x, train_y, test_y = train_test_split(segments, y, test_size=settings.test_share,
                                                        random_state=int(rng.integers(2 ** 31)))
    pca = PCA(n_components=min(settings.components, *train_x.shape), random_state=int(rng.integers(2 ** 31)))
    train_x, test_x = pca.fit_transform(train_x), pca.transform(test_x)
    train_x, train_y = smote(train_x, train_y, settings.smote_neighbours, rng)
    tree = DecisionTreeClassifier(random_state=int(rng.integers(2 ** 31))).fit(train_x, train_y)
    return float((tree.predict(test_x) == test_y).mean())


def select_channels(x, y, settings: ChannelSelectionSettings, seed: int):
    """(the k selected channel indices in ascending order, how often each was among the best k)."""
    rng = np.random.default_rng(seed)
    k = min(settings.channels, x.shape[1])
    counts = Counter()
    for _ in range(settings.repetitions):
        accuracies = [channel_accuracy(x, y, c, settings, rng) for c in range(x.shape[1])]
        best = sorted(range(x.shape[1]), key=lambda c: accuracies[c], reverse=True)[:k]   # stable: ties by index
        counts.update(best)
    chosen = [c for c, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]]
    return sorted(chosen), {int(c): int(n) for c, n in counts.items()}
