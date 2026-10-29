"""MSCFormer on version 27's samples, with both of version 27's protocols.

The paper evaluates within subject (training session, five-fold split for
validation, test session). Here the test samples are version 27's, and the
five-fold split is made on the training side of each test fold, as for
M-ShallowConvNet:

  Leave one subject out: the other subjects in five groups (validation by subject).
  Within subject: the nine training folds in five parts, trials kept together.

Scores of a subject: each of the five models is scored on all the subject's
test samples (within subject: over its 10 test folds), and the five models'
metrics are averaged, as the paper averages its five test results. For the
ROC curve the five models' mean probability is stored, with its own metrics
under "<score>_mean_probability".
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ...datascripts.splits import split_groups
from ...datascripts.version27 import CLASS_NAMES, SUBJECTS, fold_indices
from ...evaluation.complexity import environment, model_complexity
from ...evaluation.metrics import METRICS, classification_metrics, format_metrics, summarize
from .network import build_model
from .preparation import load_subject, z_score
from .settings import ModelSettings, TrainingSettings
from .training import SCORES, run_fold

PROTOCOLS = ("loso", "within")
PAPER = "MSCFormer (Zhao et al. 2025, Sci. Rep. 15, 12935)"
SCORE_DESCRIPTIONS = dict(
    validation_selected="The paper's protocol: five models, each at its lowest validation loss over {epochs} epochs; "
                        "their test metrics are averaged. The validation sets come from the training side of the "
                        "fold. No test label is used.",
    final_epoch="The five models after epoch {epochs}, metrics averaged: the protocol of version 27. No test label "
                "is used.",
    highest_test_epoch="Each model at the epoch with its highest test accuracy, chosen with the test labels, metrics "
                       "averaged; biased upward and recorded only for comparison.")


def save_models(states, folder: Path, name: str, dataset: str, channels: int, samples: int, branch_dropout: float,
                model_settings: ModelSettings) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for m, state in enumerate(states, start=1):
        torch.save(dict(state_dict=state, dataset=dataset, channels=channels, samples=samples,
                        branch_dropout=branch_dropout, model_settings=asdict(model_settings)),
                   folder / f"{name}_model{m}.pt")


def subject_scores(folds) -> dict:
    """The subject's scores from its test folds (one fold for leave one subject out)."""
    labels = [v for fold in folds for v in fold["labels"]]
    models = len(folds[0]["members"])
    result = dict(labels=labels)
    for name in SCORES:
        per_model = []
        for m in range(models):
            probabilities = [v for fold in folds for v in fold["members"][m][f"{name}_probabilities"]]
            per_model.append(classification_metrics(labels, [int(p > 0.5) for p in probabilities], probabilities))
        averaged = {key: float(np.mean([r[key] for r in per_model])) for key in METRICS if per_model[0][key] is not None}
        averaged["confusion_matrix"] = np.mean([r["confusion_matrix"] for r in per_model], axis=0).tolist()
        result[name] = dict(averaged, accuracy_per_model=[r["accuracy"] for r in per_model])
        mean_probability = np.mean([[v for fold in folds for v in fold["members"][m][f"{name}_probabilities"]]
                                    for m in range(models)], axis=0)
        result[f"{name}_scores"] = mean_probability.tolist()
        result[f"{name}_predictions"] = [int(p > 0.5) for p in mean_probability]
        result[f"{name}_mean_probability"] = classification_metrics(labels, result[f"{name}_predictions"],
                                                                    result[f"{name}_scores"])
    return result


def run(data_dir, dataset: str, protocol: str = "loso", subjects=None, device=None,
        training: TrainingSettings = TrainingSettings(), model_settings: ModelSettings = ModelSettings(),
        show_progress: bool = True, models_dir: Path | None = None) -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    cohort = SUBJECTS[dataset]
    if subjects and not set(subjects) <= set(cohort):
        raise ValueError(f"Subjects must be among the {dataset} subjects of version 27: {list(cohort)}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branch_dropout = model_settings.branch_dropout_cross if protocol == "loso" else model_settings.branch_dropout_within
    data = {}
    for subject in (cohort if protocol == "loso" else subjects or cohort):
        print(f"S{subject}: preparing samples", flush=True)
        data[subject] = load_subject(data_dir, dataset, subject)
    results = {}
    for subject in subjects or cohort:
        x, y, groups = data[subject]
        if protocol == "loso":
            others = [s for s in cohort if s != subject]
            train_x = np.concatenate([data[s][0] for s in others])
            train_y = np.concatenate([data[s][1] for s in others])
            owner = np.concatenate([[s] * len(data[s][1]) for s in others])
            parts = split_groups(others, training.models, training.seed, subject)
            splits = [(np.flatnonzero(~np.isin(owner, part)), np.flatnonzero(np.isin(owner, part))) for part in parts]
            train_x, test_x = z_score(train_x, x)
            record, states = run_fold((train_x, train_y), test_x, y, splits, branch_dropout, device, training,
                                      model_settings, f"Held-out S{subject}", show_progress)
            record["validation_subjects"] = parts
            if models_dir is not None:
                save_models(states, models_dir, f"S{subject}", dataset, x.shape[1], x.shape[2], branch_dropout,
                            model_settings)
            results[str(subject)] = dict(subject_scores([record]), **record)
        else:
            records = []
            for k, (train_index, test_index) in enumerate(fold_indices(groups, y)):
                parts = split_groups(groups[train_index], training.models, training.seed, 1000 * subject + k)
                splits = [(np.flatnonzero(~np.isin(groups[train_index], part)),
                           np.flatnonzero(np.isin(groups[train_index], part))) for part in parts]
                train_x, test_x = z_score(x[train_index], x[test_index])
                record, states = run_fold((train_x, y[train_index]), test_x, y[test_index], splits, branch_dropout,
                                          device, training, model_settings, f"S{subject} fold {k + 1}/10",
                                          show_progress)
                records.append(dict(fold=k + 1, test_index=test_index.tolist(), **record))
                if models_dir is not None:
                    save_models(states, models_dir, f"S{subject}_fold{k + 1}", dataset, x.shape[1], x.shape[2],
                                branch_dropout, model_settings)
            results[str(subject)] = dict(subject_scores(records), folds=records)
        fold = results[str(subject)]
        minutes = sum(f["training_seconds"] for f in fold.get("folds", [fold])) / 60
        print(f"S{subject} ({minutes:.1f} min): validation-selected {format_metrics(fold['validation_selected'])}\n"
              f"     final epoch          {format_metrics(fold['final_epoch'])}\n"
              f"     highest test epoch   {format_metrics(fold['highest_test_epoch'])}", flush=True)
    summary = {name: summarize({s: r[name] for s, r in results.items()}) for name in SCORES}
    print(f"\nMean over {len(results)} subjects (each subject: the mean of its five models)")
    for name in SCORES:
        print(f"  {name.replace('_', ' '):<20} {format_metrics(summary[name])}", flush=True)

    folds = [f for r in results.values() for f in r.get("folds", [r])]
    seconds = [f["training_seconds"] for f in folds]
    memory = [f["peak_gpu_memory_bytes"] for f in folds if f["peak_gpu_memory_bytes"] is not None]
    cost = dict(training_seconds_total=sum(seconds), training_seconds_per_fold_mean=float(np.mean(seconds)),
                training_seconds_per_fold_min=min(seconds), training_seconds_per_fold_max=max(seconds),
                epochs_per_fold=training.epochs, models_per_fold=training.models,
                peak_gpu_memory_bytes=max(memory) if memory else None,
                note=f"All {training.models} models of a fold, {training.epochs} epochs each, including the "
                     f"evaluation of the validation and test samples after every epoch. Data preparation is not "
                     f"included.")
    sample_x = data[next(iter(data))][0]
    complexity = model_complexity(lambda: build_model(sample_x.shape[1], sample_x.shape[2],
                                                      branch_dropout=branch_dropout, settings=model_settings),
                                  sample_x, device, training.batch_size)
    return dict(paper=PAPER, model="MSCFormer", dataset=dataset, class_names=list(CLASS_NAMES[dataset]),
                protocol="leave_one_subject_out" if protocol == "loso" else "within_subject_cross_validation",
                subjects=list(results), samples="version 27's samples, subjects and folds (datascripts/version27.py)",
                scores=list(SCORES), main_score="validation_selected",
                evaluation={name: text.format(epochs=training.epochs) for name, text in SCORE_DESCRIPTIONS.items()},
                dropout=branch_dropout, summary=summary, results=results, model_settings=asdict(model_settings),
                training=asdict(training), complexity=complexity, cost=cost, **environment(device))
