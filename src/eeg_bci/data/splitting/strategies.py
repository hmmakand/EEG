"""Split strategy names and normalized methodology groups."""

from __future__ import annotations

SESSION_TRAIN_TEST = "session_train_test"
SESSION_TRAIN_VALID_TEST = "session_train_valid_test"
SESSION_CROSS_VALIDATION_TEST = "session_cross_validation_test"
SESSION_GRID_SEARCH_TEST = "session_grid_search_test"
SESSION_LEAVE_ONE_SUBJECT_OUT = "session_leave_one_subject_out"
CHRONOLOGICAL_TRAIN_TEST = "chronological_train_test"
CHRONOLOGICAL_TRAIN_VALID_TEST = "chronological_train_valid_test"
CHRONOLOGICAL_CROSS_VALIDATION_TEST = "chronological_cross_validation_test"
CHRONOLOGICAL_GRID_SEARCH_TEST = "chronological_grid_search_test"
CHRONOLOGICAL_LEAVE_ONE_SUBJECT_OUT = "chronological_leave_one_subject_out"

TRAIN_TEST = "train_test"
TRAIN_VALID_TEST = "train_valid_test"
CROSS_VALIDATION_TEST = "cross_validation_test"
GRID_SEARCH_TEST = "grid_search_test"
LOSO = "loso"

CHRONOLOGICAL_SPLIT_STRATEGIES = {
    CHRONOLOGICAL_TRAIN_TEST,
    CHRONOLOGICAL_TRAIN_VALID_TEST,
    CHRONOLOGICAL_CROSS_VALIDATION_TEST,
    CHRONOLOGICAL_GRID_SEARCH_TEST,
}

SESSION_SPLIT_STRATEGIES = {
    SESSION_TRAIN_TEST,
    SESSION_TRAIN_VALID_TEST,
    SESSION_CROSS_VALIDATION_TEST,
    SESSION_GRID_SEARCH_TEST,
}

LOSO_SPLIT_STRATEGIES = {
    SESSION_LEAVE_ONE_SUBJECT_OUT,
    CHRONOLOGICAL_LEAVE_ONE_SUBJECT_OUT,
}

STRATEGY_METHODS = {
    SESSION_TRAIN_TEST: TRAIN_TEST,
    SESSION_TRAIN_VALID_TEST: TRAIN_VALID_TEST,
    SESSION_CROSS_VALIDATION_TEST: CROSS_VALIDATION_TEST,
    SESSION_GRID_SEARCH_TEST: GRID_SEARCH_TEST,
    SESSION_LEAVE_ONE_SUBJECT_OUT: LOSO,
    CHRONOLOGICAL_TRAIN_TEST: TRAIN_TEST,
    CHRONOLOGICAL_TRAIN_VALID_TEST: TRAIN_VALID_TEST,
    CHRONOLOGICAL_CROSS_VALIDATION_TEST: CROSS_VALIDATION_TEST,
    CHRONOLOGICAL_GRID_SEARCH_TEST: GRID_SEARCH_TEST,
    CHRONOLOGICAL_LEAVE_ONE_SUBJECT_OUT: LOSO,
}


def split_method(strategy: str) -> str:
    """Return the common training/evaluation methodology for a strategy."""

    if strategy == "description":
        strategy = SESSION_TRAIN_TEST
    try:
        return STRATEGY_METHODS[strategy]
    except KeyError as exc:
        available = ", ".join(sorted(STRATEGY_METHODS))
        raise ValueError(
            f"Unsupported split strategy {strategy}. Available: {available}."
        ) from exc
