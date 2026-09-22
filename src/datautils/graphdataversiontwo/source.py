"""Read local Liu2024 EDF recordings and decode their shared event schedule."""

from pathlib import Path
import re

import mne
import numpy as np
import pandas as pd


def discover_recordings(source_root: str | Path) -> dict[int, Path]:
    """Find one local EDF recording per subject without downloading data.

    Parameters
    ----------
    source_root : str or pathlib.Path
        The Liu2024 dataset root, its ``files`` directory, or ``edffile``.

    Returns
    -------
    dict of int to pathlib.Path
        Absolute EDF paths indexed by subject number, in ascending order.

    Raises
    ------
    FileNotFoundError
        If the directory contains no expected local EDF recordings.
    ValueError
        If a subject has multiple recordings or an unexpected filename.

    Notes
    -----
    Source files are only inspected; no files are changed or downloaded.
    """
    source_root = Path(source_root).expanduser().resolve()
    candidates = (source_root / "files" / "edffile", source_root / "edffile", source_root)
    paths = []
    for candidate in candidates:
        paths = sorted(candidate.glob("sub-*/eeg/*.edf"))
        if paths:
            break
    if not paths:
        raise FileNotFoundError(f"No local Liu2024 EDF recordings beneath {source_root}")
    recordings = {}
    for path in paths:
        match = re.fullmatch(r"sub-(\d+)", path.parents[1].name)
        if match is None:
            raise ValueError(f"Unexpected subject directory for {path}")
        subject = int(match.group(1))
        if path.name != f"sub-{subject:02d}_task-motor-imagery_eeg.edf":
            raise ValueError(f"Unexpected Liu2024 recording filename: {path}")
        if subject in recordings:
            raise ValueError(f"Multiple recordings found for subject {subject}")
        recordings[subject] = path
    return dict(sorted(recordings.items()))


def load_recording(path: str | Path) -> mne.io.BaseRaw:
    """Open an EDF lazily with the same channel scaling as the notebook.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing local Liu2024 EDF filename.

    Returns
    -------
    mne.io.BaseRaw
        Non-preloaded recording. EEG values returned by ``get_data`` are
        volts; the unnamed marker channel is also physically scaled by EDF.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.

    Notes
    -----
    This does not call MOABB download APIs or modify the source recording.
    The unnamed channel is deliberately read exactly as in the notebook.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local Liu2024 EDF file is missing: {path}")
    return mne.io.read_raw_edf(path, preload=False, verbose="ERROR")


def _find_events_path(source_path: Path) -> Path:
    """Locate the shared events TSV alongside the extracted EDF directory.

    Parameters
    ----------
    source_path : pathlib.Path
        Absolute EDF path inside the local Liu2024 directory tree.

    Returns
    -------
    pathlib.Path
        Existing Figshare file ``38516084`` or its ``.tsv`` copy.

    Raises
    ------
    FileNotFoundError
        If neither expected event file exists beside ``edffile``.

    Notes
    -----
    File contents and names are left unchanged.
    """
    for ancestor in source_path.parents:
        if ancestor.name == "edffile":
            for name in ("38516084", "38516084.tsv"):
                candidate = ancestor.parent / name
                if candidate.is_file():
                    return candidate
            break
    raise FileNotFoundError(f"Cannot locate Liu2024 events file 38516084 for {source_path}")


def read_trial_table(
    raw: mne.io.BaseRaw,
    subject: int,
    source_path: str | Path,
    *,
    events_path: str | Path | None = None,
) -> pd.DataFrame:
    """Align EDF marker stages with the source schedule and decode MI labels.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Uncropped EDF as returned by :func:`load_recording`, containing its
        unnamed marker channel. EEG data need not be preloaded.
    subject : int
        Subject identifier from the EDF's ``sub-XX`` directory.
    source_path : str or pathlib.Path
        Source EDF path, recorded in every row for provenance.
    events_path : str or pathlib.Path, optional
        Shared source events TSV. By default, locate ``38516084`` beside
        the source recording's ``edffile`` ancestor.

    Returns
    -------
    pandas.DataFrame
        One row per MI trial in source order. Includes integer ``subject``,
        zero-based ``trial_index``, ``target`` (left_hand/right_hand), integer
        ``label`` (0/1), inclusive ``start_sample``, exclusive ``stop_sample``,
        ``sampling_frequency_hz``, ``source_file``, source ``trial_type`` and
        zero-based ``source_event_row``. Sample offsets refer to ``raw``.

    Raises
    ------
    ValueError
        If trial boundaries or class codes cannot be unambiguously matched
        to the source schedule, or the input has been cropped.
    FileNotFoundError
        If the local shared events file cannot be found.

    Notes
    -----
    Installed MOABB's ``Liu2024.encoding`` maps ``trial_type=1`` to left
    hand and ``trial_type=2`` to right hand for ``value=2`` (MI). EDF stage
    marker 2 identifies MI onset, not its class. Labels are read from the
    TSV, never inferred from alternating trial indices. The TSV's timing
    describes the common protocol in milliseconds; actual EDF markers
    determine each subject's sample boundaries. Each trial has a code-1
    instruction anchor at its protocol-block onset. The unique code-2 to
    code-3 pair in that block must have the nominal MI length plus at most
    two samples of observed marker timing jitter. Extra pulses after the
    block's break and nonintegral EDF marker fill are recorded in
    ``result.attrs["marker_audit"]``; ambiguous trials raise an error.
    No input is modified.
    """
    source_path = Path(source_path).expanduser().resolve()
    if not isinstance(subject, (int, np.integer)) or isinstance(subject, bool) or subject < 1:
        raise ValueError("subject must be a positive integer")
    if source_path.parents[1].name != f"sub-{subject:02d}":
        raise ValueError(f"Subject {subject} does not match source recording {source_path}")
    if raw.first_samp != 0:
        raise ValueError("read_trial_table requires the uncropped source recording")
    events_path = _find_events_path(source_path) if events_path is None else Path(events_path)
    schedule = pd.read_csv(events_path, sep="\t")
    required = {"onset", "duration", "value", "trial_type"}
    if not required.issubset(schedule.columns):
        raise ValueError(f"Events TSV must contain {sorted(required)}")
    numeric = schedule[list(sorted(required))].to_numpy(dtype=float)
    if not len(schedule) or not np.isfinite(numeric).all():
        raise ValueError("Events TSV must contain finite event fields")
    if not np.array_equal(numeric, np.rint(numeric)):
        raise ValueError("Events TSV must contain integral timing, stage, and class fields")
    expected_codes = schedule["value"].to_numpy(dtype=np.int64)
    trial_types = schedule["trial_type"].to_numpy(dtype=np.int64)
    if not np.isin(expected_codes, (1, 2, 3)).all() or not np.isin(trial_types, (1, 2)).all():
        raise ValueError("Unsupported stage or trial_type code in source events TSV")

    marker_picks = [index for index, name in enumerate(raw.ch_names) if not name.strip()]
    if len(marker_picks) != 1:
        raise ValueError(f"Expected one unnamed marker channel; found {len(marker_picks)}")
    physical_markers = np.asarray(raw.get_data(picks=marker_picks)[0], dtype=np.float64) * 1e6
    if not np.isfinite(physical_markers).all():
        raise ValueError(f"Non-finite marker values for subject {subject}")
    rounded_markers = np.rint(physical_markers).astype(np.int64)
    integral = np.isclose(physical_markers, rounded_markers, rtol=0, atol=1e-3)
    # EDF trailing fill can be a fractional constant; rounding it creates
    # nonexistent MI/break events. Only genuine integral marker values count.
    marker_codes = np.where(integral, rounded_markers, 0)
    if not np.isin(marker_codes, (0, 1, 2, 3)).all():
        raise ValueError(f"Unknown marker codes for subject {subject}: {np.unique(marker_codes)}")
    onsets = np.flatnonzero((marker_codes != 0) & (marker_codes != np.r_[0, marker_codes[:-1]]))
    mi_rows = np.flatnonzero(expected_codes == 2)
    if not len(mi_rows) or not np.array_equal(expected_codes, np.tile([1, 2, 3], len(mi_rows))):
        raise ValueError("Source schedule must contain instruction/MI/break triplets")
    if not np.all(trial_types.reshape(-1, 3) == trial_types.reshape(-1, 3)[:, :1]):
        raise ValueError("Source schedule changes hand class within a trial")
    sfreq = float(raw.info["sfreq"])
    schedule_onsets = schedule["onset"].to_numpy(dtype=float)
    schedule_durations = schedule["duration"].to_numpy(dtype=float)
    if np.any(schedule_durations <= 0) or not np.array_equal(np.diff(schedule_onsets), schedule_durations[:-1]):
        raise ValueError("Source schedule must have contiguous positive-duration stages")
    # These relative protocol anchors are present at 0, 4000, ..., 156000
    # samples in the local EDFs. They also disambiguate subject 43's two
    # code-2 pulses that occur after a break, before the next trial starts.
    block_starts = np.rint((schedule_onsets[::3] - schedule_onsets[0]) * sfreq / 1000).astype(np.int64)
    protocol_stop = round((schedule_onsets[-1] + schedule_durations[-1] - schedule_onsets[0]) * sfreq / 1000)
    if protocol_stop > raw.n_times or not np.all(marker_codes[block_starts] == 1):
        raise ValueError(f"Subject {subject}: missing expected protocol instruction anchors")
    rows = []
    consumed_markers = set()
    ignored_markers = []
    for trial_index, event_row in enumerate(mi_rows):
        block_start = int(block_starts[trial_index])
        block_stop = int(block_starts[trial_index + 1]) if trial_index + 1 < len(mi_rows) else protocol_stop
        block_onsets = onsets[(onsets >= block_start) & (onsets < block_stop)]
        mi_starts = block_onsets[marker_codes[block_onsets] == 2]
        break_starts = block_onsets[marker_codes[block_onsets] == 3]
        expected_length = round(schedule_durations[event_row] * sfreq / 1000.0)
        candidates = [
            (int(start), int(stop))
            for start in mi_starts for stop in break_starts
            if expected_length <= stop - start <= expected_length + 2
            and not np.any((block_onsets > start) & (block_onsets < stop))
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Subject {subject}, trial {trial_index}: expected one unambiguous "
                f"code-2-to-code-3 pair in [{block_start}, {block_stop}); found {candidates}"
            )
        start, stop = candidates[0]
        used = {block_start, start, stop}
        consumed_markers.update(used)
        for onset in block_onsets:
            if int(onset) not in used:
                if onset <= stop:
                    raise ValueError(f"Subject {subject}, trial {trial_index}: unexpected marker at {onset} before trial end")
                ignored_markers.append({
                    "sample": int(onset), "code": int(marker_codes[onset]),
                    "reason": "extra marker after this protocol block's MI endpoint",
                    "trial_index": trial_index,
                })
        trial_type = int(trial_types[event_row])
        rows.append({
            "subject": int(subject),
            "trial_index": trial_index,
            "target": {1: "left_hand", 2: "right_hand"}[trial_type],
            "label": trial_type - 1,
            "start_sample": start,
            "stop_sample": stop,
            "sampling_frequency_hz": sfreq,
            "source_file": str(source_path),
            "trial_type": trial_type,
            "source_event_row": int(event_row),
        })
    if np.any(onsets >= protocol_stop):
        raise ValueError(f"Subject {subject}: unexpected integer markers beyond source protocol")
    result = pd.DataFrame(rows)
    fractional_indices = np.flatnonzero(~integral)
    fractional_ranges = []
    if fractional_indices.size:
        groups = np.split(fractional_indices, np.flatnonzero(np.diff(fractional_indices) > 1) + 1)
        for group in groups:
            fractional_ranges.append({
                "start_sample": int(group[0]), "stop_sample": int(group[-1]) + 1,
                "values": np.unique(physical_markers[group]).tolist(),
            })
    result.attrs["marker_audit"] = {
        "events_file": str(Path(events_path).resolve()),
        "integer_tolerance": 1e-3,
        "mi_duration_extra_samples_allowed": 2,
        "policy": "Unique nominal-duration code-2/code-3 pair inside each source protocol block; code-1 anchor required",
        "integer_marker_counts": {str(code): int(np.count_nonzero(marker_codes[onsets] == code)) for code in (1, 2, 3)},
        "used_marker_count": len(consumed_markers),
        "ignored_integer_markers": ignored_markers,
        "nonintegral_marker_sample_count": int(len(fractional_indices)),
        "nonintegral_marker_ranges": fractional_ranges,
        "mi_duration_sample_counts": {str(int(length)): int(count) for length, count in zip(*np.unique(result.stop_sample - result.start_sample, return_counts=True))},
    }
    return result
