"""Fixed, versioned settings for the inspected Liu2024 graph representation."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / 'data/moabb/MNE-liu2024-data'
OUTPUT_ROOT = REPO_ROOT / 'data/moabb/Graph-Liu2024-VersionTwo'
SCHEMA_VERSION = 2
CHANNEL_NAMES = (
    'FP1', 'FP2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4',
    'FT7', 'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CP3', 'CP4', 'TP7',
    'TP8', 'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2',
)
BAND_NAMES = ('delta', 'theta', 'alpha', 'beta', 'gamma')
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 40))
NODE_FEATURE_NAMES = tuple(f'{band}_power_db' for band in BAND_NAMES) + ('spectral_entropy', 'hjorth_mobility', 'hjorth_complexity')
NODE_VARIANTS = ('without_csd', 'csd')
METHODS = ('wpli', 'plv', 'icoh_abs')
CLASS_MAPPING = {'left_hand': 0, 'right_hand': 1}
EXPECTED_TRIALS_PER_SUBJECT = 40
EXPECTED_SFREQ = 500.0
EXPECTED_TRIAL_SAMPLE_RANGE = (2000, 2002)
ANALYSIS_MONTAGE_NAME = 'standard_1020'
CSD_SETTINGS = {'lambda2': 1e-5, 'stiffness': 4, 'n_legendre_terms': 50}
PSD_WINDOW_SECONDS = 2.0
PSD_OVERLAP = 0.5
POWER_FLOOR = 1e-12
FREQUENCIES = tuple(range(1, 41))
CONNECTIVITY_SETTINGS = {
    'mode': 'multitaper', 'sm_times': 0.5, 'mt_bandwidth': 2.0,
    'n_cycles': 3.0, 'average': False, 'faverage': False,
}
