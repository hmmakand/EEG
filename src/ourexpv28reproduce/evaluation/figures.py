"""Figures and the model table from a result file of a reproduced paper.

    python -m src.ourexpv28reproduce.evaluation.figures RESULT.json [RESULT.json ...]

For every result file, a folder <name>_figures/ next to it with
    roc.png               ROC curve over all test trials, final epoch and as published
    training_curves.png   loss and accuracy per epoch, mean over folds (source batches and held-out subject)
    calibration.png       share of target trials with a calibrated target, and how many are right
    confusion_matrix.png  summed over subjects, final epoch
    subjects.png          accuracy of every subject, final epoch and as published
    model.txt             parameters, FLOPs, inference time, training time, memory, software
With several result files, roc_comparison.png in the folder of the first compares their final-epoch ROC curves.

The final epoch is chosen without test labels. "As published" is the epoch chosen with the test labels, as the
paper's code does; it is shown for comparison with the paper.
"""
import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

SCORES = (("final_epoch", "final epoch (chosen without test labels)", "-"),
          ("as_published", "as published (chosen with test labels)", "--"))


def pooled(result, key):
    return np.concatenate([fold[key] for fold in result["results"].values()])


def title_of(result):
    return f"{result['paper'].split(' (')[0]}, {result['dataset']}, cohort {result['cohort']}, leave one subject out"


def roc_figure(result, path):
    labels = pooled(result, "labels")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for key, name, style in SCORES:
        fpr, tpr, _ = roc_curve(labels, pooled(result, f"{key}_scores"))
        ax.plot(fpr, tpr, style, label=f"{name}, AUC {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ":", color="grey", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title(title_of(result), fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def mean_std(result, key):
    with warnings.catch_warnings():                      # epochs without any value (e.g. before calibration)
        warnings.simplefilter("ignore", RuntimeWarning)
        values = np.array([[np.nan if v is None else v for v in fold[key]] for fold in result["results"].values()],
                          dtype=float)
        return np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def training_figure(result, path):
    epochs = np.arange(1, result["training"]["epochs"] + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, kind in zip(axes, ("loss", "accuracy")):
        for key, name in ((f"train_{kind}_per_epoch", "source batches (training)"),
                          (f"test_{kind}_per_epoch", "held-out subject")):
            mean, std = mean_std(result, key)
            ax.plot(epochs, mean, label=name)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)
        ax.axvline(result["settings"]["warmup_epochs"] + 0.5, color="grey", lw=0.8, ls=":")
        ax.set(xlabel="Epoch", ylabel=kind.capitalize())
        ax.legend(fontsize=8)
    fig.suptitle(f"{title_of(result)}: mean ± SD over {len(result['results'])} folds", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def calibration_figure(result, path):
    epochs = np.arange(1, result["training"]["epochs"] + 1)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for key, name in (("calibrated_fraction_per_epoch", "target trials with a target"),
                      ("target_accuracy_per_epoch", "targets that are right")):
        ax.plot(epochs, mean_std(result, key)[0], label=name)
    ax.axvline(result["settings"]["warmup_epochs"] + 0.5, color="grey", lw=0.8, ls=":")
    ax.set(xlabel="Epoch", ylabel="Share", ylim=(0, 1.02))
    ax.set_title(f"PGTC ({result['pgtc']}), {result['dataset']}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_figure(result, path):
    matrix = np.array(result["summary"]["final_epoch"]["confusion_matrix"])
    names = ["class 0", "class 1"]
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{value}\n{value / matrix[row].sum():.1%}", ha="center", va="center", fontsize=9,
                color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=(0, 1), yticks=(0, 1), xticklabels=names, yticklabels=names, xlabel="Predicted", ylabel="True")
    ax.set_title(f"{result['dataset']}, final epoch", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def subjects_figure(result, path):
    subjects = list(result["results"])
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(max(5, 0.25 * len(subjects) + 2), 3.4))
    for offset, (key, name, _) in zip((-0.2, 0.2), SCORES):
        values = [result["results"][s][key]["accuracy"] for s in subjects]
        ax.bar(x + offset, values, 0.4, alpha=1.0 if key == "final_epoch" else 0.5,
               label=f"{name}, mean {np.mean(values):.1%}")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set(xticks=x, xlabel="Subject", ylabel="Accuracy", ylim=(0, 1))
    ax.set_xticklabels(subjects, fontsize=7)
    ax.set_title(title_of(result), fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def model_text(result) -> str:
    size, cost = result["complexity"], result["cost"]
    memory = cost["peak_gpu_memory_bytes"]
    rows = [("Device", f"{result['device_name']} (CUDA {result['cuda']})"),
            ("Parameters", f"{size['parameters']:,}"),
            ("FLOPs per trial", f"{size['flops_per_trial'] / 1e6:.2f} M"),
            ("MACs per trial", f"{size['macs_per_trial'] / 1e6:.2f} M"),
            ("Model file size", f"{size['model_file_bytes'] / 1024:.1f} KiB"),
            ("Inference, one trial alone", f"{size['latency_ms_per_trial']:.2f} ms"),
            (f"Inference, per trial in a batch of {size['latency_batch_size']}",
             f"{size['latency_ms_per_trial_in_batch']:.3f} ms"),
            ("Training time, all folds", f"{cost['training_seconds_total'] / 3600:.2f} h"),
            (f"Training time per fold of {cost['epochs_per_fold']} epochs (mean, min-max)",
             f"{cost['training_seconds_per_fold_mean']:.1f} s ({cost['training_seconds_per_fold_min']:.1f}-"
             f"{cost['training_seconds_per_fold_max']:.1f})"),
            ("Peak GPU memory in training", f"{memory / 2 ** 20:.0f} MiB" if memory else "not measured (CPU)"),
            ("Python", result["python"]),
            ("Packages", ", ".join(f"{k} {v}" for k, v in result["packages"].items()))]
    width = max(len(name) for name, _ in rows)
    lines = [title_of(result), ""] + [f"{name:<{width}}  {value}" for name, value in rows]
    return "\n".join(lines + ["", size["flop_note"], size["latency_note"], cost["note"]]) + "\n"


def comparison_figure(results, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for name, result in results:
        fpr, tpr, _ = roc_curve(pooled(result, "labels"), pooled(result, "final_epoch_scores"))
        ax.plot(fpr, tpr, label=f"{name}, AUC {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC, final epoch")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args(argv)
    loaded = []
    for path in args.results:
        result = json.loads(path.read_text())
        loaded.append((path.stem, result))
        folder = path.with_name(path.stem + "_figures")
        folder.mkdir(exist_ok=True)
        roc_figure(result, folder / "roc.png")
        training_figure(result, folder / "training_curves.png")
        calibration_figure(result, folder / "calibration.png")
        confusion_figure(result, folder / "confusion_matrix.png")
        subjects_figure(result, folder / "subjects.png")
        (folder / "model.txt").write_text(model_text(result))
        print(f"Figures of {path.name} saved to {folder}")
    if len(loaded) > 1:
        target = args.results[0].with_name(args.results[0].stem + "_figures") / "roc_comparison.png"
        comparison_figure(loaded, target)
        print(f"Comparison saved to {target}")


if __name__ == "__main__":
    main()
