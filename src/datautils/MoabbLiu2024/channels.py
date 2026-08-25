"""EEG-channel selection for Liu2024 recordings."""

from typing import Any

from braindecode.preprocessing import Preprocessor, preprocess


def select_eeg_channels(dataset: Any, *, n_jobs: int = 1) -> Any:
    """Modify a Braindecode dataset in place so that it contains EEG only."""

    preprocess(
        dataset,
        [Preprocessor("pick", picks="eeg", apply_on_array=False)],
        n_jobs=n_jobs,
    )
    return dataset
