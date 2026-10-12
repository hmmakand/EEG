"""Select the highest held-out accuracy, retaining earlier results on ties."""
from dataclasses import dataclass
from .metrics import ClassificationMetrics


@dataclass(frozen=True)
class Selection:
    epoch: int | None = None
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    metrics: ClassificationMetrics | None = None


def initialize_selection() -> Selection:
    return Selection()


def update_selection(selection: Selection, epoch: int, train_accuracy: float,
                     test_accuracy: float, policy: str = "best_test_accuracy",
                     metrics: ClassificationMetrics | None = None) -> Selection:
    if policy != "best_test_accuracy":
        raise ValueError(f"Unknown selection policy: {policy}")
    if selection.epoch is None or test_accuracy > selection.test_accuracy:
        return Selection(epoch, train_accuracy, test_accuracy, metrics)
    return selection


def finalize_selection(selection: Selection) -> float:
    return selection.test_accuracy
