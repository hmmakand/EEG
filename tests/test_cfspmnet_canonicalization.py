from __future__ import annotations

import numpy as np
import pytest

from eeg_bci.cfspmnet.canonicalization import (
    LIU2024_EEG_CHANNELS,
    LIU2024_FLIP_PAIRS,
    build_flip_indices,
    to_affected_unaffected,
)


def test_build_flip_indices_swaps_known_pairs() -> None:
    flip_indices = build_flip_indices()
    for left, right in LIU2024_FLIP_PAIRS:
        left_idx = LIU2024_EEG_CHANNELS.index(left)
        right_idx = LIU2024_EEG_CHANNELS.index(right)
        assert flip_indices[left_idx] == right_idx
        assert flip_indices[right_idx] == left_idx


def test_build_flip_indices_keeps_midline_channels_fixed() -> None:
    flip_indices = build_flip_indices()
    flip_pair_channels = {name for pair in LIU2024_FLIP_PAIRS for name in pair}
    midline_channels = [ch for ch in LIU2024_EEG_CHANNELS if ch not in flip_pair_channels]
    assert midline_channels, "expected at least one untouched midline channel"
    for name in midline_channels:
        idx = LIU2024_EEG_CHANNELS.index(name)
        assert flip_indices[idx] == idx


def test_build_flip_indices_is_an_involution() -> None:
    flip_indices = build_flip_indices()
    assert np.array_equal(flip_indices[flip_indices], np.arange(len(LIU2024_EEG_CHANNELS)))


def test_build_flip_indices_actually_reorders_a_trial() -> None:
    flip_indices = build_flip_indices()
    x = np.arange(len(LIU2024_EEG_CHANNELS) * 3, dtype=np.float32).reshape(len(LIU2024_EEG_CHANNELS), 3)
    flipped = x[flip_indices]
    fc3_idx = LIU2024_EEG_CHANNELS.index("FC3")
    fc4_idx = LIU2024_EEG_CHANNELS.index("FC4")
    assert np.array_equal(flipped[fc3_idx], x[fc4_idx])
    assert np.array_equal(flipped[fc4_idx], x[fc3_idx])


@pytest.mark.parametrize(
    ("hand_target", "paralysis_side", "expected"),
    [
        (0, "left", 0),  # left_hand + left paralysis -> affected
        (1, "left", 1),  # right_hand + left paralysis -> unaffected
        (0, "right", 1),  # left_hand + right paralysis -> unaffected
        (1, "right", 0),  # right_hand + right paralysis -> affected
    ],
)
def test_to_affected_unaffected_remaps_by_paralysis_side(hand_target, paralysis_side, expected) -> None:
    assert to_affected_unaffected(hand_target, paralysis_side) == expected


def test_to_affected_unaffected_rejects_invalid_hand_target() -> None:
    with pytest.raises(ValueError, match="Expected target 0 or 1"):
        to_affected_unaffected(2, "left")


def test_to_affected_unaffected_rejects_invalid_paralysis_side() -> None:
    with pytest.raises(ValueError, match="Unsupported paralysis side"):
        to_affected_unaffected(0, "both")
