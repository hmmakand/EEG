"""JSON metadata for reproducing the two-node/six-edge Liu2024 dataset."""

import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def file_sha256(path):
    """Hash a local file without loading it entirely into memory.

    Parameters
    ----------
    path : path-like
        File to read without modification.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_montage_metadata(eeg, source_root):
    """Describe unchanged source coordinates and the attached analysis geometry.

    Parameters
    ----------
    eeg : mne.io.BaseRaw
        EEG with the analysis montage attached, in saved channel order.
    source_root : path-like
        Local MNE-liu2024-data directory containing files/38516078.

    Returns
    -------
    dict
        JSON-compatible source coordinates (units/frame undeclared), analysis
        positions in head coordinates/meters, fiducials and acquisition details.

    Raises
    ------
    ValueError
        If source coordinates do not cover the EEG channels or are invalid.
    """
    path = Path(source_root) / 'files/38516078'
    if not path.is_file():
        path = path.with_suffix('.tsv')
    table = pd.read_csv(path, sep='\t')
    if not {'name', 'X', 'Y', 'Z'} <= set(table.columns):
        raise ValueError(f'Invalid electrode TSV columns: {path}')
    xyz = table[['X', 'Y', 'Z']].to_numpy(dtype=float)
    if table['name'].isna().any() or table['name'].duplicated().any() or not np.isfinite(xyz).all():
        raise ValueError(f'Invalid electrode names/positions: {path}')
    source_positions = dict(zip(table['name'], xyz.tolist()))
    if not set(eeg.ch_names) <= source_positions.keys():
        raise ValueError('Source electrode positions do not cover all EEG channels')
    montage = eeg.get_montage()
    if montage is None:
        raise ValueError('Analysis montage is missing')
    positions = montage.get_positions()
    return {
        'acquisition_system': '10-10', 'acquisition_reference': 'CPz', 'ground': 'FPz',
        'channel_names': list(eeg.ch_names),
        'source_electrodes': {
            'path': str(path.relative_to(source_root)),
            'url': 'https://ndownloader.figshare.com/files/38516078',
            'sha256': file_sha256(path), 'units': None, 'coordinate_frame': None,
            'note': 'Units and coordinate frame are not declared in the source TSV.',
            'channel_positions': source_positions,
        },
        'analysis_montage': {
            'kind': 'standard_template', 'name': config.ANALYSIS_MONTAGE_NAME,
            'units': 'm', 'coordinate_frame': positions['coord_frame'],
            'channel_positions': {ch: positions['ch_pos'][ch].tolist() for ch in eeg.ch_names},
            'fiducials': {key: None if positions[key] is None else positions[key].tolist()
                          for key in ('nasion', 'lpa', 'rpa')},
        },
    }


def generation_settings():
    """Return fixed calculation settings, units and feature conventions.

    Returns
    -------
    dict
        JSON-compatible settings; no files or inputs are modified.
    """
    return {
        'schema_version': config.SCHEMA_VERSION,
        'channel_names': list(config.CHANNEL_NAMES),
        'bands_hz': dict(zip(config.BAND_NAMES, config.BANDS)),
        'band_boundaries': 'Both nominal endpoints included explicitly.',
        'band_averaging_correction': 'mne-connectivity 0.9.0 _foi_average doubles the upper index for multi-bin bands. Use faverage=False and average the requested frequency bins explicitly.',
        'preprocessing': {
            'source': 'Locally supplied EDF, read directly by MNE without downloading.',
            'additional_temporal_filter': None, 'resampling': None,
            'additional_re_reference_without_csd': None, 'artifact_rejection': None,
            'normalization': None, 'excluded_channels': ['CPz', 'HEOL', 'HEOR', ''],
            'note': 'Source EDF may already be preprocessed; without_csd means no additional CSD.',
        },
        'trials': {'onset_marker': 2, 'stop_marker': 3, 'stop_exclusive': True,
                   'samples': 'Actual marker-defined length recorded per sample', 'sfreq_hz': 500.0,
                   'label_source': 'files/38516084 trial_type: 1=left_hand, 2=right_hand'},
        'csd': {'montage': config.ANALYSIS_MONTAGE_NAME, 'sphere': 'auto-fit recorded per subject',
                **config.CSD_SETTINGS},
        'welch': {'window': 'hann', 'segment_seconds': 2.0, 'overlap_fraction': 0.5,
                  'detrend': 'constant', 'scaling': 'density', 'average': 'mean',
                  'integration': 'scipy.integrate.simpson', 'power_floor_in_reference_units': 1e-12},
        'entropy': {'range_hz': [1, 40], 'normalized': True, 'log_base': 2,
                    'definition': 'Shannon entropy of normalized PSD bins divided by log2(number of bins)'},
        'connectivity': {'methods': ['wpli', 'plv', 'imcoh'],
                         'frequencies_hz': list(config.FREQUENCIES),
                         **config.CONNECTIVITY_SETTINGS,
                         'icoh_absolute': 'Absolute value after signed frequency-band averaging.',
                         'sm_freqs': 1, 'sm_kernel': 'hanning', 'padding': 0.0, 'decim': 1, 'n_jobs': 1,
                         'parameters_not_listed': 'Installed mne-connectivity defaults; version recorded.'},
        'node_units': {
            'without_csd': {'signal': 'V', 'linear_power': 'V^2', 'db_reference': '1 microvolt^2',
                            'db_reference_si': 1e-12, 'spectral_entropy': 'dimensionless'},
            'csd': {'signal': 'V/m^2', 'linear_power': 'V^2/m^4',
                    'db_reference': '1 (microvolt/m^2)^2', 'db_reference_si': 1e-12,
                    'spectral_entropy': 'dimensionless'},
        },
        'edges': {'self_loops': False, 'threshold': None,
                  'order': 'numpy.triu_indices(29,k=1), then reversed pairs',
                  'units': 'dimensionless', 'range': [0, 1]},
    }


def build_dataset_metadata(n_samples, subjects, montage, subject_records, arrays, fingerprint,
                           expected_trials_per_subject=40):
    """Assemble the metadata manifest for a completed generation run.

    Parameters
    ----------
    n_samples : int
        Number of rows shared by all per-trial arrays.
    subjects : sequence of int
        Included subjects in saved order.
    montage : dict
        Output of build_montage_metadata.
    subject_records : list of dict
        EDF checksums, fitted spheres and source provenance per subject.
    arrays : dict
        Mapping of filenames to shape/dtype declarations.
    fingerprint : str
        Digest identifying calculation settings, implementation and software.
    expected_trials_per_subject : int
        Required trial count for each included subject, including pilot runs.

    Returns
    -------
    dict
        JSON-compatible dataset manifest with explicit source/feature ordering.
    """
    software = {name: importlib.metadata.version(name)
                for name in ('numpy', 'scipy', 'pandas', 'mne', 'mne-connectivity')}
    variants = {f'{method}_{signal}': {
        'file': f'edge_attr_{method}_{signal}.npy', 'method': method,
        'signal_source': signal, 'band_names': list(config.BAND_NAMES),
    } for signal in config.NODE_VARIANTS for method in config.METHODS}
    return {
        'dataset': 'Liu2024', 'schema_version': config.SCHEMA_VERSION, 'status': 'complete',
        'n_samples': n_samples, 'subjects': list(subjects), 'excluded_subjects': [],
        'expected_trials_per_subject': expected_trials_per_subject,
        **({'expected_class_counts_per_subject': {'left_hand': 20, 'right_hand': 20}}
           if expected_trials_per_subject == 40 else {}),
        'channel_names': list(config.CHANNEL_NAMES), 'node_feature_names': list(config.NODE_FEATURE_NAMES),
        'band_names': list(config.BAND_NAMES), 'class_mapping': config.CLASS_MAPPING,
        'node_variants': {key: {'file': f'node_features_{key}.npy', 'signal_source': key,
                               **generation_settings()['node_units'][key]} for key in config.NODE_VARIANTS},
        'edge_variants': variants, 'arrays': arrays, 'montage': montage,
        'generation_settings': generation_settings(), 'generation_fingerprint': fingerprint,
        'software_versions': software, 'connectivity_backend': 'mne',
        'source_records': subject_records,
        'source_notebook': 'src/datautils/graphdataversionone/readmoabbliu.ipynb',
        'pilot': expected_trials_per_subject != 40 or len(subjects) != 50,
    }
