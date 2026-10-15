"""Figures and the model table from a result file of version 27.

    python -m src.ourexpv27.analysis.figures RESULT.json [RESULT.json ...] [--no-tsne]

For every result file, a folder <name>_figures/ next to it with
    roc.png               ROC curve over all test samples, final epoch and highest test epoch
    training_curves.png   loss and accuracy per epoch, mean over folds and models (train and test)
    calibration.png       share of test samples with a calibration target, and how many are right (if used)
    confusion_matrix.png  summed over subjects, final epoch
    subjects.png          accuracy of every subject, final and highest test epoch
    tsne.png              t-SNE of the test samples' features (graph head and temporal encoder), by class
    model.txt             parameters, FLOPs, inference time, training time, memory
With several result files, roc_comparison.png in the folder of the first compares their final-epoch ROC curves.

The final epoch is the result; the highest test epoch is chosen with the test labels and is shown for
comparison only.
"""
import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve


def folds_of(result):
    """Every fold of the run: the subject results for leave one subject out, the folds for within subject."""
    if result["protocol"] == "leave_one_subject_out":
        return list(result["subjects"].values())
    return [fold for subject in result["subjects"].values() for fold in subject["folds"]]


def pooled(result, key):
    return np.concatenate([fold[key] for fold in folds_of(result)])


def title_of(result):
    protocol = "leave one subject out" if result["protocol"] == "leave_one_subject_out" else "within subject"
    return f"{result['dataset']}, {result['task'].replace('_', ' ')}, {protocol}"


def roc_figure(result, path):
    labels = pooled(result, "labels")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for key, name, style in (("scores", "final epoch (result)", "-"),
                             ("highest_epoch_scores", "highest test epoch (biased)", "--")):
        fpr, tpr, _ = roc_curve(labels, pooled(result, key))
        ax.plot(fpr, tpr, style, label=f"{name}, AUC {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], ":", color="grey", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title_of(result))
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(0.98, 0.32, f"mean AUC over subjects {result['summary']['auc']:.3f}", ha="right", fontsize=8,
            transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def mean_curve(histories, key):
    curves = np.array([history[key] for history in histories if history.get(key)], dtype=float)
    return np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)


def training_figure(result, path):
    histories = [history for fold in folds_of(result) for history in fold["history"]]
    epochs = np.arange(1, len(histories[0]["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, kind in zip(axes, ("loss", "accuracy")):
        for key, name in ((f"train_{kind}", "training"), (f"test_{kind}", "test (one model)")):
            mean, std = mean_curve(histories, key)
            ax.plot(epochs, mean, label=name)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)
        if kind == "accuracy":
            ensemble = np.array([fold["test_accuracy_per_epoch"] for fold in folds_of(result)])
            ax.plot(epochs, ensemble.mean(0), "k--", label="test (ensemble, centred)")
        else:
            ensemble = np.array([fold["test_loss_per_epoch"] for fold in folds_of(result)])
            ax.plot(epochs, ensemble.mean(0), "k--", label="test (ensemble)")
        warmup = result["train_config"]["calibration_warmup_epochs"]
        if result["train_config"]["target_calibration"]:
            ax.axvline(warmup + 0.5, color="grey", lw=0.8, ls=":")
        ax.set(xlabel="Epoch", ylabel=kind.capitalize())
        ax.legend(fontsize=8)
    fig.suptitle(f"{title_of(result)}: mean ± SD over {len(histories)} models", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def calibration_figure(result, path):
    histories = [history for fold in folds_of(result) for history in fold["history"]]
    if "calibration_labelled_fraction" not in histories[0]:
        return False
    epochs = np.arange(1, len(histories[0]["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    fraction = np.array([h["calibration_labelled_fraction"] for h in histories], dtype=float)
    correct = np.array([[np.nan if v is None else v for v in h["calibration_target_accuracy"]] for h in histories],
                       dtype=float)
    ax.plot(epochs, fraction.mean(0), label="test samples with a target")
    with warnings.catch_warnings():                      # epochs before the first target are empty
        warnings.simplefilter("ignore", RuntimeWarning)
        ax.plot(epochs, np.nanmean(correct, axis=0), label="targets that are right")
    ax.axvline(result["train_config"]["calibration_warmup_epochs"] + 0.5, color="grey", lw=0.8, ls=":")
    ax.set(xlabel="Epoch", ylabel="Share", ylim=(0, 1.02))
    ax.set_title(f"Target calibration, {title_of(result)}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def confusion_figure(result, path):
    matrix = np.array(result["summary"]["confusion_matrix"])
    names = result["class_names"]
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        ax.text(column, row, f"{value}\n{value / matrix[row].sum():.1%}", ha="center", va="center", fontsize=9,
                color="white" if value > matrix.max() / 2 else "black")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
           xlabel="Predicted", ylabel="True")
    ax.set_title(title_of(result), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def subjects_figure(result, path):
    subjects = list(result["subjects"])
    final = [result["subjects"][s]["accuracy"] for s in subjects]
    highest = [result["subjects"][s]["highest_epoch"]["accuracy"] for s in subjects]
    x = np.arange(len(subjects))
    fig, ax = plt.subplots(figsize=(max(5, 0.22 * len(subjects) + 2), 3.4))
    ax.bar(x - 0.2, final, 0.4, label=f"final epoch, mean {np.mean(final):.1%}")
    ax.bar(x + 0.2, highest, 0.4, alpha=0.5, label=f"highest test epoch, mean {np.mean(highest):.1%}")
    ax.axhline(1 / len(result["class_names"]), color="grey", ls=":", lw=0.8)
    ax.set(xticks=x, xlabel="Subject", ylabel="Accuracy", ylim=(0, 1))
    ax.set_xticklabels(subjects, fontsize=7)
    ax.set_title(title_of(result), fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def tsne_figure(result, embeddings_file, path, seed=12345):
    features = np.load(embeddings_file)
    names = [name for name in ("graph", "temporal") if any(key.startswith(name + "_") for key in features)]
    if not names:
        return False
    labels = np.concatenate([np.concatenate([f["labels"] for f in subject["folds"]]) if "folds" in subject
                             else subject["labels"] for subject in result["subjects"].values()])
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        values = np.concatenate([features[f"{name}_S{s}"] for s in result["subjects"]])
        values = (values - values.mean(0)) / (values.std(0) + 1e-8)
        points = TSNE(n_components=2, random_state=seed, init="pca", perplexity=min(30, len(values) - 1)).fit_transform(values)
        for label, class_name in enumerate(result["class_names"]):
            chosen = labels == label
            ax.scatter(*points[chosen].T, s=3, alpha=0.5, label=class_name)
        ax.set_title(f"{'Graph head' if name == 'graph' else 'Temporal encoder'} features "
                     f"({values.shape[1]} dimensions)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=8, markerscale=3)
    fig.suptitle(f"t-SNE of the test samples, first model of each fold, final epoch; {title_of(result)}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def model_text(result) -> str:
    size, cost = result["complexity"], result["cost"]
    memory = cost["peak_gpu_memory_bytes"]
    rows = [("Device", result["device_name"]),
            ("Parameters (one model)", f"{size['parameters']:,}"),
            ("  graph head / wide path / temporal encoder",
             " / ".join(f"{size['parameters_per_path'][p]:,}" for p in ("graph", "wide", "temporal"))),
            ("Models per decision (ensemble)", str(size["models_per_decision"])),
            ("FLOPs per sample, one model (lower bound)", f"{size['flops_per_sample'] / 1e6:.2f} M"),
            ("MACs per sample, one model (lower bound)", f"{size['macs_per_sample'] / 1e6:.2f} M"),
            ("Model file size", f"{size['model_file_bytes'] / 1024:.1f} KiB"),
            ("Inference, one sample alone", f"{size['latency_ms_per_sample']:.2f} ms"),
            (f"Inference, per sample in a batch of {size['latency_batch_size']}",
             f"{size['latency_ms_per_sample_in_batch']:.3f} ms"),
            ("Training time, all folds", f"{cost['training_seconds_total'] / 3600:.2f} h"),
            ("Training time per fold (mean, min-max)",
             f"{cost['training_seconds_per_fold_mean']:.1f} s ({cost['training_seconds_per_fold_min']:.1f}-"
             f"{cost['training_seconds_per_fold_max']:.1f})"),
            ("Peak GPU memory in training", f"{memory / 2 ** 20:.0f} MiB" if memory else "not measured (CPU)"),
            ("Packages", ", ".join(f"{k} {v}" for k, v in result["packages"].items()))]
    width = max(len(name) for name, _ in rows)
    lines = [title_of(result), ""] + [f"{name:<{width}}  {value}" for name, value in rows]
    return "\n".join(lines + ["", size["flop_note"], size["latency_note"], cost["note"]]) + "\n"


def comparison_figure(results, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for name, result in results:
        labels, scores = pooled(result, "labels"), pooled(result, "scores")
        fpr, tpr, _ = roc_curve(labels, scores)
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
    parser.add_argument("--no-tsne", action="store_true", help="Skip the t-SNE plot (the slowest part)")
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
        embeddings_file = path.with_name(path.stem + ".embeddings.npz")
        if not args.no_tsne and embeddings_file.exists():
            tsne_figure(result, embeddings_file, folder / "tsne.png")
        (folder / "model.txt").write_text(model_text(result))
        print(f"Figures of {path.name} saved to {folder}")
    if len(loaded) > 1:
        target = args.results[0].with_name(args.results[0].stem + "_figures") / "roc_comparison.png"
        comparison_figure(loaded, target)
        print(f"Comparison saved to {target}")


if __name__ == "__main__":
    main()
