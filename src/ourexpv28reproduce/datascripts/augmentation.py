"""Segmentation and reconstruction (S&R), the time-domain augmentation of EEG Conformer and MSCFormer."""
import torch


def segment_and_reconstruct(x, y, size: int, segments: int, classes: int = 2):
    """S&R: `size` new samples, size // classes per class; segment j of a new sample comes from a random
    training sample of its class (conformer.py: interaug)."""
    length = x.shape[2]
    bounds = [length * j // segments for j in range(segments + 1)]
    new_x, new_y = [], []
    for label in range(classes):
        pool = torch.nonzero(y == label).flatten()
        count = size // classes
        sources = pool[torch.randint(len(pool), (count, segments), device=x.device)]
        parts = [x[sources[:, j], :, bounds[j]:bounds[j + 1]] for j in range(segments)]
        new_x.append(torch.cat(parts, dim=2))
        new_y.append(torch.full((count,), label, dtype=torch.long, device=x.device))
    order = torch.randperm(size // classes * classes, device=x.device)
    return torch.cat(new_x)[order], torch.cat(new_y)[order]
