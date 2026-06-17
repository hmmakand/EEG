from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    n_chans: int
    n_outputs: int
    n_times: int
    sfreq: float
