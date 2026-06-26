from __future__ import annotations

import pytest

from eeg_bci.tracking.tensorboard import write_confusion_matrix


def test_write_confusion_matrix_rejects_class_name_shape_mismatch(tmp_path) -> None:
    with pytest.raises(ValueError, match="class_names length"):
        write_confusion_matrix(tmp_path, [[1, 0], [0, 1]], class_names=["only_one"])
