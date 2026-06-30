"""Split methodology constants.

Two axes make up a split configuration:

- ``source`` -- how train_pool/test_set are carved out of the raw dataset.
  This is dataset-structure-dependent (e.g. BCI IV 2a has a ``session``
  column, Liu2024 doesn't and uses ``chronological`` ordering instead).
  Sources are registered in ``splitting/sources.py``'s ``SOURCE_BUILDERS``.
- ``method`` -- the dataset-agnostic training/evaluation methodology applied
  on top of a (train_pool, test_set) pair (``TRAIN_TEST``,
  ``TRAIN_VALID_TEST``, ``CROSS_VALIDATION_TEST``, ``GRID_SEARCH_TEST``,
  ``LOSO``).

A dataset's ``split`` config sets both directly, e.g.::

    split:
      source: session
      method: grid_search_test

Adding a new dataset-structure-dependent split mechanism only requires
registering a new source builder in ``splitting/sources.py`` -- the five
methods above already work with any source.
"""

from __future__ import annotations

SOURCE_SESSION = "session"
SOURCE_CHRONOLOGICAL = "chronological"
SOURCE_RANDOM = "random"

TRAIN_TEST = "train_test"
TRAIN_VALID_TEST = "train_valid_test"
CROSS_VALIDATION_TEST = "cross_validation_test"
GRID_SEARCH_TEST = "grid_search_test"
LOSO = "leave_one_subject_out"

METHODS = {
    TRAIN_TEST,
    TRAIN_VALID_TEST,
    CROSS_VALIDATION_TEST,
    GRID_SEARCH_TEST,
    LOSO,
}


def split_label(source: str, method: str) -> str:
    """Human-readable label combining a split source and method."""

    return f"{source}_{method}"
