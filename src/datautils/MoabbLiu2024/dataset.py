"""End-to-end public interface for preparing Liu2024."""

from .channels import select_eeg_channels
from .config import Liu2024Config
from .loader import load_liu2024
from .preprocessing import preprocess_liu2024
from .validation import validate_liu2024
from .windowing import create_left_right_windows


def prepare_liu2024_dataset(config: Liu2024Config | None = None):
    """Load, validate, preprocess, and window the configured Liu2024 subjects."""

    config = config or Liu2024Config()
    dataset = load_liu2024(config.subject_ids, config.data_root)
    validate_liu2024(dataset)
    select_eeg_channels(dataset, n_jobs=config.n_jobs)
    validate_liu2024(dataset, require_eeg_only=True)
    preprocess_liu2024(
        dataset,
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        target_sfreq=config.target_sfreq,
        standardize=config.standardize,
        factor_new=config.standardize_factor_new,
        init_block_size=config.standardize_init_block_size,
        n_jobs=config.n_jobs,
    )
    validate_liu2024(
        dataset,
        require_eeg_only=True,
        expected_sfreq=config.target_sfreq,
    )
    return create_left_right_windows(
        dataset,
        event_mapping=config.event_mapping,
        trial_start_offset_seconds=config.trial_start_offset_seconds,
        trial_stop_offset_seconds=config.trial_stop_offset_seconds,
        preload=config.preload_windows,
        n_jobs=config.n_jobs,
    )
