from __future__ import annotations

import numpy as np

from eeg_bci.data.windowing import infer_window_info, n_outputs_from_mapping


def _windows(targets: list[int]) -> list[tuple[np.ndarray, int, None]]:
    return [(np.zeros((3, 10), dtype=np.float32), int(t), None) for t in targets]


def test_n_outputs_from_mapping_uses_head_size() -> None:
    assert n_outputs_from_mapping({"a": 0, "b": 1, "c": 2, "d": 3}) == 4
    assert n_outputs_from_mapping({"left_hand": 0, "right_hand": 1}) == 2


def test_n_outputs_from_mapping_none_when_absent() -> None:
    assert n_outputs_from_mapping(None) is None
    assert n_outputs_from_mapping({}) is None


def test_infer_window_info_prefers_explicit_n_outputs() -> None:
    # Only classes 0 and 1 are observed, but the mapping declares 4.
    windows = _windows([0, 1, 0, 1])

    n_chans, n_outputs, n_times = infer_window_info(windows, n_outputs=4)

    assert (n_chans, n_outputs, n_times) == (3, 4, 10)


def test_infer_window_info_falls_back_to_observed() -> None:
    windows = _windows([0, 1, 2])

    _, n_outputs, _ = infer_window_info(windows)

    assert n_outputs == 3
