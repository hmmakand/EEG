"""Per-subject canonicalization for cross-subject Liu2024 stroke MI-EEG.

Every Liu2024 subject records ``left_hand``/``right_hand`` MI, but which hand
is paretic (affected) differs per subject (see
``data/moabb/MNE-liu2024-data/files/participants.tsv``). Two steps make
subjects comparable for cross-subject (LOSO) modeling, following
``temp/sppm_strategy.py``'s ``canonicalize_subject_trials``:

1. **Label remap** -- ``left_hand``/``right_hand`` -> ``affected``/``unaffected``,
   using each subject's ``ParalysisSide``.
2. **Channel flip** -- for left-paralysis subjects, swap left/right channel
   pairs (e.g. ``FC3``<->``FC4``) so "affected hemisphere" is always spatially
   on the same side across subjects. This is what makes SPPM's private-signature
   channel groups (fixed left/right motor triplets, see ``sppm.py``) meaningful
   across subjects with different paralysis sides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Captured from `build_moabb_dataset(subject_ids=[1])` on the project's
# configured Liu2024 preprocessing pipeline (`configs/preprocessing/liu2024.yaml`):
# EEG-only channels, in-order, after MOABB's Liu2024 loader drops the
# reference electrode (CPz) and EOG/stim channels.
LIU2024_EEG_CHANNELS: tuple[str, ...] = (
    "FP1", "FP2", "Fz", "F3", "F4", "F7", "F8", "FCz", "FC3", "FC4",
    "FT7", "FT8", "Cz", "C3", "C4", "T3", "T4", "CP3", "CP4", "TP7",
    "TP8", "Pz", "P3", "P4", "T5", "T6", "Oz", "O1", "O2",
)

# Channel order in Figshare's raw ``sourcedata.zip`` MAT files. Unlike the
# processed EDF used by MOABB, this representation retains the CPz reference.
LIU2024_FIGSHARE_EEG_CHANNELS: tuple[str, ...] = (
    *LIU2024_EEG_CHANNELS[:17],
    "CPz",
    *LIU2024_EEG_CHANNELS[17:],
)

# Left/right symmetric 10-20 pairs actually present in LIU2024_EEG_CHANNELS.
# Midline channels (Fz, FCz, Cz, Pz, Oz) are left untouched.
LIU2024_FLIP_PAIRS: tuple[tuple[str, str], ...] = (
    ("FP1", "FP2"), ("F3", "F4"), ("F7", "F8"), ("FC3", "FC4"),
    ("FT7", "FT8"), ("C3", "C4"), ("T3", "T4"), ("CP3", "CP4"),
    ("TP7", "TP8"), ("P3", "P4"), ("T5", "T6"), ("O1", "O2"),
)

LEFT_HAND, RIGHT_HAND = 0, 1
AFFECTED, UNAFFECTED = 0, 1


def load_liu2024_participants(participants_path: Path | str) -> pd.DataFrame:
    """Load and validate `participants.tsv`, indexed by integer MOABB subject id."""

    participants_path = Path(participants_path)
    if not participants_path.exists():
        raise FileNotFoundError(f"Liu2024 participants file not found: {participants_path}")

    participants = pd.read_csv(participants_path, sep="\t")
    participants["subject"] = (
        participants["Participant_ID"].str.extract(r"^sub-(\d+)$", expand=False).astype(int)
    )
    if not participants["subject"].is_unique:
        raise ValueError("participants.tsv has duplicate subject ids.")
    if not set(participants["ParalysisSide"]).issubset({"left", "right"}):
        raise ValueError("participants.tsv has an unexpected ParalysisSide value.")
    return participants.set_index("subject", drop=False)


def to_affected_unaffected(hand_target: int, paralysis_side: str) -> int:
    """Remap a left/right-hand MI label to affected/unaffected using paralysis side."""

    if hand_target not in {LEFT_HAND, RIGHT_HAND}:
        raise ValueError(f"Expected target 0 or 1, got {hand_target}")
    if paralysis_side == "left":
        return AFFECTED if hand_target == LEFT_HAND else UNAFFECTED
    if paralysis_side == "right":
        return AFFECTED if hand_target == RIGHT_HAND else UNAFFECTED
    raise ValueError(f"Unsupported paralysis side: {paralysis_side!r}")


def build_flip_indices(channel_names: tuple[str, ...] = LIU2024_EEG_CHANNELS) -> np.ndarray:
    """Index array that swaps each left/right channel pair, leaving midline channels fixed."""

    swap_map: dict[str, str] = {}
    for left, right in LIU2024_FLIP_PAIRS:
        swap_map[left] = right
        swap_map[right] = left
    missing = [name for pair in LIU2024_FLIP_PAIRS for name in pair if name not in channel_names]
    if missing:
        raise ValueError(f"Flip-pair channels missing from channel_names: {missing}")
    return np.asarray(
        [channel_names.index(swap_map.get(name, name)) for name in channel_names],
        dtype=np.int64,
    )


def canonicalize_subject_trials(
    x: np.ndarray,
    y: np.ndarray,
    subject_id: int,
    participants: pd.DataFrame,
    channel_names: tuple[str, ...] = LIU2024_EEG_CHANNELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the channel-flip + label-remap canonicalization for one subject's trials.

    Parameters
    ----------
    x : np.ndarray, shape (n_trials, n_chans, n_times)
    y : np.ndarray, shape (n_trials,) of raw 0/1 left/right-hand labels.
    """

    if subject_id not in participants.index:
        raise ValueError(f"Subject {subject_id} is missing from participants.tsv.")

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if x.ndim != 3:
        raise ValueError(f"Expected trials with shape (N, C, T), got {x.shape}")

    paralysis_side = str(participants.loc[subject_id, "ParalysisSide"])
    canonical_x = x.copy()
    if paralysis_side == "left":
        canonical_x = canonical_x[:, build_flip_indices(channel_names), :]

    canonical_y = np.asarray(
        [to_affected_unaffected(int(target), paralysis_side) for target in y], dtype=np.int64
    )
    return canonical_x, canonical_y
