"""Display and persist results without calculating metrics."""
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_results_path(connectivity_method: str, output_path: Path | None = None, *, threshold: float) -> Path:
    """Honor explicit paths; otherwise identify connectivity and graph threshold in the filename."""
    if output_path is not None:
        return Path(output_path)
    return (Path(__file__).resolve().parents[1] / "output" /
            f"BCI_IV_2a_GAT_{connectivity_method.upper()}_Threshold_{threshold}_Results.json")


def print_subject_result(subject_number: int, results: Mapping[str, Any]) -> None:
    print(f"S{subject_number}: Mean: {results['mean']:.4f}, Max: {results['max']:.4f}, Min: {results['min']:.4f}")

    if "balanced_accuracy" in results:
        for name, label in (("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro F1")):
            values = results[name]
            print(f"  {label}: Mean: {values['mean']:.4f}, Max: {values['max']:.4f}, Min: {values['min']:.4f}")
        print(f"  Confusion matrix (rows=true, columns=predicted; labels={results['class_labels']}):")
        for row in results["confusion_matrix"]:
            print(f"    {row}")


def print_summary(subject_results: Mapping[int, Mapping[str, Any]]) -> None:
    print("\nSummary of Results for All Subjects:")
    for subject_number, results in subject_results.items():
        print_subject_result(subject_number, results)


def save_results(subject_results: Mapping[int, Mapping[str, Any]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as json_file:
        json.dump(subject_results, json_file, indent=4)
    print(f"\nResults saved to '{output_path}'")
