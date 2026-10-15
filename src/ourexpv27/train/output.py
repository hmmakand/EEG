"""Where results and models go: one folder per dataset inside the output directory.

    <output-dir>/<dataset>/Version27_<dataset>_<Task>_<Protocol>_seed<seed>.json
    <output-dir>/<dataset>/Version27_<dataset>_<Task>_<Protocol>_seed<seed>.summary.txt
    <output-dir>/<dataset>/Version27_<dataset>_<Task>_<Protocol>_seed<seed>.embeddings.npz
    <output-dir>/<dataset>/models/<name of the result file>/...           with --save-models
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

TASK_NAMES = dict(liu2024="ImageryRest", bci2a="LeftRight", jia2019="LeftRight")


def result_path(output_dir, dataset: str, protocol: str, seed: int, variant: str = "") -> Path:
    """A result file that does not exist yet; a time stamp is added if the plain name is taken.
    `variant` names an ablation, e.g. "NoCalibration"; it is added to the name."""
    folder = Path(output_dir) / dataset
    suffix = f"_{variant}" if variant else ""
    path = folder / f"Version27_{dataset}_{TASK_NAMES[dataset]}_{protocol}{suffix}_seed{seed}.json"
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = path.with_name(f"{path.stem}_{stamp}.json")
    return path


def models_folder(result_file: Path) -> Path:
    return result_file.parent / "models" / result_file.stem


def summary_text(result: dict) -> str:
    """The table of the result file in readable form: every subject, then the mean, for the final and the
    highest test epoch."""
    from .metrics import format_metrics, METRICS
    lines = [f"{result['dataset']}, {result['task']}, {result['protocol']}, seed {result['train_config']['seed']}",
             "Final epoch: the result. Highest test epoch: chosen with the test labels, for comparison only.", ""]
    for subject, scores in result["subjects"].items():
        lines.append(f"S{subject:<3} final epoch          {format_metrics(scores)}")
        lines.append(f"     highest test epoch   {format_metrics(scores['highest_epoch'])}")
    summary = result["summary"]
    lines += ["", f"Mean over {len(result['subjects'])} subjects",
              f"     final epoch          {format_metrics({name: summary[name] for name in METRICS})}",
              f"     highest test epoch   {format_metrics(summary['highest_epoch'])}"]
    model = result["model_config"]
    if tuple(model["paths"]) != ("graph", "wide", "temporal") or not result["train_config"]["target_calibration"]:
        lines += ["", f"Ablation: paths {'+'.join(model['paths'])}, "
                      f"target calibration {'on' if result['train_config']['target_calibration'] else 'off'}"]
    cost, size = result.get("cost"), result.get("complexity")
    if size:
        lines += ["", f"Model ({result['device_name']}): {size['parameters']:,} parameters "
                      f"(graph {size['parameters_per_path']['graph']:,}, wide {size['parameters_per_path']['wide']:,}, "
                      f"temporal {size['parameters_per_path']['temporal']:,}), "
                      f"{size['flops_per_sample'] / 1e6:.2f} MFLOPs per sample (lower bound), "
                      f"file {size['model_file_bytes'] / 1024:.0f} KiB",
                  f"Inference: {size['latency_ms_per_sample']:.2f} ms per sample alone, "
                  f"{size['latency_ms_per_sample_in_batch']:.3f} ms per sample in a batch of "
                  f"{size['latency_batch_size']}; {size['models_per_decision']} models per decision"]
    if cost:
        memory = cost["peak_gpu_memory_bytes"]
        lines.append(f"Training: {cost['training_seconds_total'] / 60:.1f} min in all, "
                     f"{cost['training_seconds_per_fold_mean']:.1f} s per fold"
                     + (f", peak GPU memory {memory / 2 ** 20:.0f} MiB" if memory else ""))
    return "\n".join(lines) + "\n"


def embeddings_path(result_file: Path) -> Path:
    return result_file.with_name(result_file.stem + ".embeddings.npz")


def write_embeddings(features: dict, path: Path) -> None:
    """One array per subject and feature: <feature>_S<subject> [samples, dimensions], in the order of the
    subject's labels in the JSON (for within subject: the pooled order, fold after fold)."""
    np.savez_compressed(path, **{f"{name}_S{subject}": values for subject, by_name in features.items()
                                 for name, values in by_name.items()})


def write_result(result: dict, path: Path) -> None:
    """The full result as JSON, the same scores as a readable table next to it (<name>.summary.txt), and
    the embeddings of the test samples (<name>.embeddings.npz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(result)
    features = result.pop("embeddings", None)
    path.write_text(json.dumps(result, indent=2) + "\n")
    table = path.with_name(path.stem + ".summary.txt")
    table.write_text(summary_text(result))
    print(f"Result saved to {path}\nSummary table saved to {table}")
    if features:
        write_embeddings(features, embeddings_path(path))
        print(f"Embeddings saved to {embeddings_path(path)}")
