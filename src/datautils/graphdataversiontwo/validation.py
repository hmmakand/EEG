"""Check trial alignment, numerical features, and saved Liu2024 files."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import NODE_FEATURE_NAMES
from .saved_dataset import EDGE_FILES, NODE_FILES, load_dataset


def validate_trial_pair(
    eeg: np.ndarray,
    csd: np.ndarray,
    *,
    n_channels: int = 29,
    expected_n_samples: int | None = None,
    context: str = "trial",
) -> None:
    """Check that EEG and CSD windows have matching, finite sample arrays.

    Parameters
    ----------
    eeg : numpy.ndarray
        EEG potentials in V, shape ``(n_channels, n_time_samples)``.
    csd : numpy.ndarray
        Surface Laplacian in V/m², with the same channel and sample order.
    n_channels : int
        Required number of channels; defaults to the 29-channel dataset.
    expected_n_samples : int or None
        Exact window length when known, otherwise only matching nonempty
        windows are required.
    context : str
        Subject/trial description included in validation errors.

    Returns
    -------
    None
        Successful validation has no return value or side effects.

    Raises
    ------
    ValueError
        If shapes differ, a window is empty, or any value is nonfinite.

    Notes
    -----
    Shape checks cannot establish semantic alignment. The caller must extract
    both arrays using one trial table and matching channel names/order.
    """
    eeg, csd = np.asarray(eeg), np.asarray(csd)
    if eeg.ndim != 2 or eeg.shape[0] != n_channels or eeg.shape[1] == 0:
        raise ValueError(f"{context}: EEG must have shape ({n_channels}, T>0); got {eeg.shape}")
    if csd.shape != eeg.shape:
        raise ValueError(f"{context}: EEG {eeg.shape} and CSD {csd.shape} windows do not match")
    if expected_n_samples is not None and eeg.shape[1] != expected_n_samples:
        raise ValueError(f"{context}: window has {eeg.shape[1]} samples; expected {expected_n_samples}")
    for name, data in (("EEG", eeg), ("CSD", csd)):
        if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
            raise ValueError(f"{context}: {name} contains nonnumeric or nonfinite values")


def _validate_edge_index(edge_index: np.ndarray, n_channels: int, context: str) -> None:
    """Check complete electrode pairs followed by their reversed directions.

    Parameters
    ----------
    edge_index : numpy.ndarray
        Integer edge endpoints, shape ``(2, n_channels * (n_channels - 1))``.
    n_channels : int
        Number of electrode nodes in the fixed channel order.
    context : str
        Description included in validation errors.

    Returns
    -------
    None
        The input is checked without modification.

    Raises
    ------
    ValueError
        If there are missing/repeated pairs, self-loops, or incorrect order.
    """
    edge_index = np.asarray(edge_index)
    first, second = np.triu_indices(n_channels, k=1)
    expected = np.concatenate((np.stack((first, second)), np.stack((second, first))), axis=1)
    if not np.issubdtype(edge_index.dtype, np.integer) or not np.array_equal(edge_index, expected):
        raise ValueError(
            f"{context}: edge_index must contain all upper-triangle channel pairs "
            "in NumPy order, followed by their reverse directions"
        )


def validate_features(
    node_features: Mapping[str, np.ndarray],
    edge_features: Mapping[str, np.ndarray],
    edge_index: np.ndarray,
    *,
    n_channels: int = 29,
    n_node_features: int = len(NODE_FEATURE_NAMES),
    n_bands: int = 5,
    context: str = "trial",
) -> None:
    """Validate both node variants and all six edge variants for one trial.

    Parameters
    ----------
    node_features : mapping of str to numpy.ndarray
        ``without_csd`` and ``csd`` arrays of shape ``(n_channels, 8)``.
        The first five columns are dB powers, followed by normalized
        entropy, Hjorth mobility and complexity. Power references/units are recorded in metadata.
    edge_features : mapping of str to numpy.ndarray
        Six method/source variants of shape ``(812, n_bands)``. All values
        are dimensionless; reversed directions must carry identical values.
    edge_index : numpy.ndarray
        Upper-triangle electrode pairs followed by reverse pairs, ``(2, 812)``.
    n_channels : int
        Expected number of nodes.
    n_node_features : int
        Expected node width; must match the configured feature contract.
    n_bands : int
        Expected number of connectivity bands.
    context : str
        Subject/trial description used to identify errors.

    Returns
    -------
    None
        Success leaves all arrays unchanged.

    Raises
    ------
    ValueError
        If a variant is missing, a shape is wrong, values are nonfinite,
        entropy/connectivity leave [0, 1], or reverse edges differ.

    Notes
    -----
    A tolerance of 1e-6 admits floating-point roundoff at range boundaries.
    Negative dB powers are valid and are not rejected.
    """
    if n_node_features != len(NODE_FEATURE_NAMES):
        raise ValueError("Node width must match the eight-feature contract")
    if set(node_features) != set(NODE_FILES):
        raise ValueError(f"{context}: node variants must be {list(NODE_FILES)}")
    if set(edge_features) != set(EDGE_FILES):
        raise ValueError(f"{context}: edge variants must be {list(EDGE_FILES)}")
    _validate_edge_index(edge_index, n_channels, context)
    n_edges = n_channels * (n_channels - 1)
    tolerance = 1e-6
    for name, values in node_features.items():
        values = np.asarray(values)
        if values.shape != (n_channels, n_node_features) or not np.isfinite(values).all():
            raise ValueError(f"{context}: node variant {name!r} has wrong shape or nonfinite values")
        entropy = values[:, NODE_FEATURE_NAMES.index("spectral_entropy")]
        if np.any(entropy < -tolerance) or np.any(entropy > 1 + tolerance):
            raise ValueError(f"{context}: node variant {name!r} spectral entropy is outside [0, 1]")
        mobility = values[:, NODE_FEATURE_NAMES.index("hjorth_mobility")]
        complexity = values[:, NODE_FEATURE_NAMES.index("hjorth_complexity")]
        if np.any(mobility <= 0) or np.any(complexity < 0):
            raise ValueError(f"{context}: node variant {name!r} has invalid Hjorth values")
    for name, values in edge_features.items():
        values = np.asarray(values)
        if values.shape != (n_edges, n_bands) or not np.isfinite(values).all():
            raise ValueError(f"{context}: edge variant {name!r} has wrong shape or nonfinite values")
        if np.any(values < -tolerance) or np.any(values > 1 + tolerance):
            raise ValueError(f"{context}: edge variant {name!r} connectivity is outside [0, 1]")
        if not np.allclose(values[: n_edges // 2], values[n_edges // 2 :], atol=1e-7, rtol=1e-6):
            raise ValueError(f"{context}: edge variant {name!r} differs between reverse directions")


def _validate_montage(metadata: dict) -> None:
    """Check saved electrode positions and the analysis coordinate convention.

    Parameters
    ----------
    metadata : dict
        Manifest with channel names and source/analysis montage descriptions.

    Returns
    -------
    None
        Metadata is checked without modification.

    Raises
    ------
    ValueError
        If positions are missing, malformed, or nonfinite, or the analysis
        montage is not described in head coordinates and meters.

    Notes
    -----
    Source coordinate units/frame may remain unknown, as in the original
    electrode TSV. Only the separately saved analysis geometry requires m/head.
    """
    montage = metadata.get("montage")
    if not isinstance(montage, dict):
        raise ValueError("metadata montage is missing")
    names = metadata["channel_names"]
    if montage.get("channel_names") != names:
        raise ValueError("montage channel order differs from saved channel_names")
    analysis = montage.get("analysis_montage", {})
    if analysis.get("units") != "m" or analysis.get("coordinate_frame") != "head":
        raise ValueError("analysis_montage must use meters and the head coordinate frame")
    for section in ("source_electrodes", "analysis_montage"):
        positions = montage.get(section, {}).get("channel_positions")
        if not isinstance(positions, dict) or not set(names) <= positions.keys():
            raise ValueError(f"montage {section} must include positions for all saved channels")
        if section == "analysis_montage" and set(positions) != set(names):
            raise ValueError("analysis_montage positions must match the saved channel names")
        for name, coordinates in positions.items():
            try:
                xyz = np.asarray(coordinates, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"montage {section} channel {name}: invalid coordinates") from exc
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"montage {section} channel {name}: expected three finite coordinates")
            if section == "analysis_montage" and np.linalg.norm(xyz) == 0:
                raise ValueError(f"analysis_montage channel {name}: electrode position is at the origin")
    for name in ("nasion", "lpa", "rpa"):
        coordinates = analysis.get("fiducials", {}).get(name)
        if coordinates is None:
            raise ValueError(f"analysis_montage fiducial {name} is missing")
        xyz = np.asarray(coordinates, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"analysis_montage fiducial {name}: expected three finite coordinates")
    for name, description in metadata.get("node_variants", {}).items():
        expected_signal_units = {"without_csd": "V", "csd": "V/m^2"}
        if name in expected_signal_units:
            if description.get("signal") != expected_signal_units[name]:
                raise ValueError(f"node variant {name}: incorrect signal units")
            reference = description.get("db_reference_si")
            if not isinstance(reference, (int, float)) or not np.isfinite(reference) or reference <= 0:
                raise ValueError(f"node variant {name}: dB reference must be positive and finite")


def validate_saved_dataset(
    path: str | Path,
    *,
    expected_trials_per_subject: int | None = None,
) -> dict:
    """Reload and fully validate feature files and their ordered trial records.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory containing saved arrays, ``samples.tsv``, and metadata.
    expected_trials_per_subject : int or None
        Required count for each saved subject. If omitted, use metadata's
        ``expected_trials_per_subject`` when present. It is 40 for complete
        Liu2024 subject recordings.

    Returns
    -------
    dict
        Number of samples, number of subjects, subject counts, and class
        counts. Dictionary keys are strings for direct JSON serialization.

    Raises
    ------
    ValueError
        If file schemas, array values, record alignment, window boundaries,
        class mapping, subject trial indices, or expected counts disagree.

    Notes
    -----
    Arrays remain memory mapped and are checked one trial at a time. This
    reads all features but modifies no files. It verifies saved alignment,
    not the scientific correctness of the upstream feature estimators.
    """
    dataset = load_dataset(path)
    metadata = dataset.metadata
    _validate_montage(metadata)
    _validate_edge_index(dataset.edge_index, len(metadata["channel_names"]), str(path))
    if expected_trials_per_subject is None:
        expected_trials_per_subject = metadata.get("expected_trials_per_subject")
    if expected_trials_per_subject is not None and expected_trials_per_subject < 1:
        raise ValueError("expected_trials_per_subject must be positive")
    class_mapping = metadata["class_mapping"]
    subject_counts = Counter()
    class_counts = Counter()
    subject_class_counts = {}
    seen_trials = set()
    for index, sample in enumerate(dataset.samples):
        subject, trial_index = sample["subject"], sample["trial_index"]
        context = f"subject {subject}, trial {trial_index}, sample {index}"
        if sample["sample_index"] != index:
            raise ValueError(f"{context}: samples.tsv sample_index does not match array row")
        if subject < 1 or trial_index < 0 or (subject, trial_index) in seen_trials:
            raise ValueError(f"{context}: invalid or duplicate subject/trial identifier")
        seen_trials.add((subject, trial_index))
        label = int(dataset.labels[index])
        if label not in class_mapping.values() or sample["label"] != label:
            raise ValueError(f"{context}: label is invalid or differs from labels.npy")
        if class_mapping.get(sample["target"]) != label:
            raise ValueError(f"{context}: target name differs from the saved class mapping")
        sfreq = sample["sampling_frequency_hz"]
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError(f"{context}: sampling_frequency_hz must be finite and positive")
        if sample["start_sample"] < 0 or sample["stop_sample"] <= sample["start_sample"]:
            raise ValueError(f"{context}: trial sample boundaries must define a nonempty window")
        if not sample["source_file"]:
            raise ValueError(f"{context}: source_file is missing")
        expected_window_samples = metadata.get("trial_n_samples")
        if expected_window_samples is not None and sample["stop_sample"] - sample["start_sample"] != expected_window_samples:
            raise ValueError(f"{context}: window length differs from metadata trial_n_samples")
        validate_features(
            {name: values[index] for name, values in dataset.nodes.items()},
            {name: values[index] for name, values in dataset.edges.items()},
            dataset.edge_index,
            context=context,
        )
        subject_counts[subject] += 1
        class_counts[sample["target"]] += 1
        subject_class_counts.setdefault(subject, Counter())[sample["target"]] += 1

    if "subjects" in metadata and set(metadata["subjects"]) != set(subject_counts):
        raise ValueError("Saved subjects differ from metadata subjects")
    for subject, count in subject_counts.items():
        if expected_trials_per_subject is not None and count != expected_trials_per_subject:
            raise ValueError(f"subject {subject}: saved {count} trials; expected {expected_trials_per_subject}")
        indices = {trial for saved_subject, trial in seen_trials if saved_subject == subject}
        if indices != set(range(count)):
            raise ValueError(f"subject {subject}: trial_index must cover 0 through {count - 1}")
        expected_classes = metadata.get("expected_class_counts_per_subject")
        if expected_classes is not None and dict(subject_class_counts[subject]) != expected_classes:
            raise ValueError(f"subject {subject}: class counts disagree with metadata")
    return {
        "n_samples": len(dataset),
        "n_subjects": len(subject_counts),
        "subject_counts": {str(key): value for key, value in sorted(subject_counts.items())},
        "class_counts": dict(sorted(class_counts.items())),
        "subject_class_counts": {
            str(subject): dict(sorted(counts.items()))
            for subject, counts in sorted(subject_class_counts.items())
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate a saved dataset directory and print a JSON count report.

    Parameters
    ----------
    argv : sequence of str or None
        Command-line arguments. The optional positional path defaults to
        ``config.OUTPUT_ROOT``. None reads the process's arguments.

    Returns
    -------
    int
        Zero after successful validation. Validation errors propagate and
        produce a nonzero process exit status when run as a module.

    Notes
    -----
    All feature arrays are read, but no dataset files are modified.
    """
    from .config import OUTPUT_ROOT

    parser = argparse.ArgumentParser(description="Validate saved Liu2024 graph features and trial alignment.")
    parser.add_argument("path", nargs="?", type=Path, default=OUTPUT_ROOT, help="Saved dataset directory")
    arguments = parser.parse_args(argv)
    print(json.dumps(validate_saved_dataset(arguments.path), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
