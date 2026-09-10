"""Prepare the inspected EEG and surface-Laplacian signal variants."""

import mne
import numpy as np
from mne.bem import fit_sphere_to_headshape
from mne.preprocessing import compute_current_source_density

from .config import ANALYSIS_MONTAGE_NAME, CHANNEL_NAMES


def prepare_eeg(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Select the 29 usable EEG channels in the dataset's fixed node order.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Local Liu2024 recording, with EEG potentials in volts.

    Returns
    -------
    mne.io.BaseRaw
        Independent channel-selected copy in ``CHANNEL_NAMES`` order.
        Its preload state and sampling frequency match the input.

    Raises
    ------
    ValueError
        If expected channels are absent, unexpected channels are present,
        or any retained channel is not EEG or is marked bad.

    Notes
    -----
    CPz (the zero reference channel), HEOL, HEOR, and the unnamed stage
    channel are excluded. No additional filtering, resampling, reference
    change, artifact correction, or normalization is applied. The input
    object is not modified.
    """
    excluded = {"CPz", "HEOL", "HEOR", ""}
    missing = set(CHANNEL_NAMES) - set(raw.ch_names)
    unexpected = {name for name in raw.ch_names if name.strip()} - set(CHANNEL_NAMES) - excluded
    if missing or unexpected:
        raise ValueError(f"Unexpected Liu2024 channels: missing={sorted(missing)}, extra={sorted(unexpected)}")
    eeg = raw.copy().pick(list(CHANNEL_NAMES))
    if tuple(eeg.ch_names) != tuple(CHANNEL_NAMES) or set(eeg.get_channel_types()) != {"eeg"}:
        raise ValueError("Selected EEG channels must match the configured node order and EEG type")
    if eeg.info["bads"]:
        raise ValueError(f"Bad EEG channels require an explicit handling policy: {eeg.info['bads']}")
    return eeg


def attach_analysis_montage(eeg: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Attach the inspected standard_1020 template for spatial analysis.

    Parameters
    ----------
    eeg : mne.io.BaseRaw
        EEG-only recording in the configured 29-channel order.

    Returns
    -------
    mne.io.BaseRaw
        The same object, with template electrode positions in meters in
        the MNE head coordinate frame and template fiducials attached.

    Raises
    ------
    ValueError
        If the channel order or template electrode positions are invalid.

    Notes
    -----
    This modifies ``eeg.info`` in place but does not change signal values.
    Case-insensitive matching preserves source names FP1 and FP2. The
    original dataset coordinates have undeclared units/frame and must be
    saved separately as provenance; they are not treated as head meters.
    """
    if tuple(eeg.ch_names) != tuple(CHANNEL_NAMES):
        raise ValueError("Analysis montage requires the configured EEG channel order")
    eeg.set_montage(
        mne.channels.make_standard_montage(ANALYSIS_MONTAGE_NAME),
        match_case=False,
        on_missing="raise",
        verbose="ERROR",
    )
    positions = eeg.get_montage().get_positions()
    xyz = np.asarray([positions["ch_pos"][name] for name in CHANNEL_NAMES])
    if positions["coord_frame"] != "head" or not np.isfinite(xyz).all() or np.any(np.linalg.norm(xyz, axis=1) == 0):
        raise ValueError("Analysis montage must provide finite nonzero positions in the head frame")
    return eeg


def compute_csd(
    eeg: mne.io.BaseRaw,
    *,
    lambda2: float = 1e-5,
    stiffness: float = 4.0,
    n_legendre_terms: int = 50,
) -> tuple[mne.io.BaseRaw, tuple[float, float, float, float]]:
    """Compute the spherical-spline surface Laplacian without changing EEG.

    Parameters
    ----------
    eeg : mne.io.BaseRaw
        EEG potentials in volts, in configured channel order, with an
        attached analysis montage. The input may be lazily loaded.
    lambda2 : float, default 1e-5
        MNE's dimensionless spline regularization parameter.
    stiffness : float, default 4
        MNE's dimensionless spherical-spline stiffness.
    n_legendre_terms : int, default 50
        Number of Legendre terms used by MNE.

    Returns
    -------
    csd : mne.io.BaseRaw
        Independent preloaded recording in V/m², with matching channels,
        samples, and time origin. The input data and preload state remain
        unchanged.
    sphere : tuple of float
        Fitted ``(x, y, z, radius)`` in meters, for saved provenance.

    Raises
    ------
    ValueError
        If channel order, montage, or CSD settings are invalid. MNE also
        rejects bad EEG channels and an already CSD-transformed input.

    Notes
    -----
    Sphere fitting and origin subtraction follow MNE's ``sphere='auto'``
    implementation exactly. The explicit fitted sphere is passed to MNE
    so the geometry can be recorded. Only the new copy is loaded/modified.
    """
    if tuple(eeg.ch_names) != tuple(CHANNEL_NAMES):
        raise ValueError("CSD requires the configured EEG channel order")
    if eeg.get_montage() is None:
        raise ValueError("Attach the analysis montage before computing CSD")
    radius, origin_head, origin_device = fit_sphere_to_headshape(eeg.info, verbose="ERROR")
    sphere = tuple(float(value) for value in (*tuple(origin_head - origin_device), radius))
    if not np.isfinite(sphere).all() or sphere[3] <= 0:
        raise ValueError(f"Invalid fitted CSD sphere: {sphere}")
    csd = compute_current_source_density(
        eeg.copy().load_data(verbose="ERROR"),
        sphere=sphere,
        lambda2=lambda2,
        stiffness=stiffness,
        n_legendre_terms=n_legendre_terms,
        copy=False,
        verbose="ERROR",
    )
    return csd, sphere


def extract_trial_pair(
    eeg: mne.io.BaseRaw,
    csd: mne.io.BaseRaw,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract matching EEG and CSD windows for a single graph sample.

    Parameters
    ----------
    eeg : mne.io.BaseRaw
        Original EEG potentials in volts in configured channel order.
    csd : mne.io.BaseRaw
        CSD signals in V/m² with the same channels and sample coordinates.
    start : int
        Inclusive code-2 onset sample, relative to these recordings.
    stop : int
        Exclusive following code-3 onset sample.

    Returns
    -------
    eeg_trial : numpy.ndarray, shape (29, stop - start)
        EEG samples in volts, in the configured channel order.
    csd_trial : numpy.ndarray, shape (29, stop - start)
        Matching CSD samples in V/m².

    Raises
    ------
    ValueError
        If recordings are misaligned, indices are invalid, or extracted
        data contain non-finite values.

    Notes
    -----
    Neither recording is modified. CSD channel types are ``csd`` in MNE,
    so explicit positional extraction preserves every configured node.
    """
    if tuple(eeg.ch_names) != tuple(CHANNEL_NAMES) or eeg.ch_names != csd.ch_names:
        raise ValueError("EEG and CSD channels must share the configured node order")
    if eeg.n_times != csd.n_times or eeg.first_samp != csd.first_samp or eeg.info["sfreq"] != csd.info["sfreq"]:
        raise ValueError("EEG and CSD recordings are not sample-aligned")
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) for value in (start, stop)):
        raise ValueError("Trial bounds must be integer sample indices")
    if not 0 <= start < stop <= eeg.n_times:
        raise ValueError(f"Invalid trial bounds [{start}, {stop}) for {eeg.n_times} samples")
    eeg_trial = eeg.get_data(start=int(start), stop=int(stop))
    csd_trial = csd.get_data(start=int(start), stop=int(stop))
    if eeg_trial.shape != csd_trial.shape or not np.isfinite(eeg_trial).all() or not np.isfinite(csd_trial).all():
        raise ValueError("EEG/CSD trial pair must have identical shapes and finite values")
    return eeg_trial, csd_trial
