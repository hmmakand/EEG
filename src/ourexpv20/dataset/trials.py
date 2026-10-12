"""Trial selection, label mapping, and ordering rules."""
from dataclasses import dataclass
import numpy as np
from .validation import validate_session, validate_trial


@dataclass(frozen=True)
class Trial:
    signal: np.ndarray  # [samples, channels]
    label: int
    subject_id: int
    session_id: int
    trial_id: int


def select_sessions(sessions, indices):
    by_id = {session.session_id: session for session in sessions}
    for index in indices:
        if index not in by_id:
            raise ValueError(f"Missing session index {index}")
        yield by_id[index]


def select_channels(signal, indices):
    if max(indices) >= signal.shape[1]:
        raise ValueError("Selected channel is outside the recording")
    return signal[:, list(indices)]


def event_to_sample(position, offset=0):
    return int(position) + offset


def encode_label(source_label, label_map):
    return dict(label_map).get(int(source_label))


def extract_trials(sessions, subject_id, sample_rate, config):
    trials = []
    length = int(config.duration_seconds * sample_rate)
    for session in select_sessions(sessions, config.session_indices):
        validate_session(session)
        signal = select_channels(session.signal, config.channel_indices)
        for index, (position, source_label) in enumerate(zip(session.positions, session.labels)):
            label = encode_label(source_label, config.label_map)
            if label is None:
                continue
            start = event_to_sample(position, config.event_index_offset)
            stop = start + length
            if start < 0 or stop > len(signal):
                if config.incomplete_policy == "skip":
                    continue
                raise ValueError(f"Incomplete trial {index} in session {session.session_id}")
            trial = Trial(signal[start:stop], label, subject_id, session.session_id, index)
            validate_trial(trial)
            trials.append(trial)
    # Original feature/connectivity arrays group all left trials, then right.
    return sorted(trials, key=lambda trial: trial.label)


def order_trials(items, policy):
    """Accept Trials or PyG Data; preserve the legacy ordering when requested."""
    if policy == "grouped":
        return list(items)
    if policy == "legacy_interleave":
        half = len(items) // 2
        return [item for pair in zip(items[:half], items[half:2 * half]) for item in pair]
    if policy == "interleave":
        groups = {}
        for item in items:
            label = item.label if isinstance(item, Trial) else int(item.y)
            groups.setdefault(label, []).append(item)
        return [group[i] for i in range(max(map(len, groups.values()), default=0))
                for _, group in sorted(groups.items()) if i < len(group)]
    raise ValueError(f"Unknown trial order: {policy}")
