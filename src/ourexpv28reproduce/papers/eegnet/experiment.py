"""EEGNet on version 27's samples, with both of version 27's protocols.

Leave one subject out: every subject is the test subject once. As in the
paper's cross-subject SMR analysis, the other subjects are split at random
into a training set and a validation set (3 of 8 there; the same share here),
and the dropout rate is 0.25.

Within subject: version 27's 10 folds of every subject are the test folds
(the same test samples as our method). For test fold k, fold k + 1 is the
validation set and the other eight folds are the training set; the dropout
rate is 0.5. The paper used 4 blocks (2 training, 1 validation, 1 test);
version 27's folds are kept so that the two methods are tested on the same
samples.
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ...datascripts.version27 import CLASS_NAMES, SUBJECTS, fold_indices
from ...evaluation.complexity import environment, model_complexity
from ...evaluation.metrics import classification_metrics, format_metrics, summarize
from .network import build_model
from .preparation import load_subject
from .settings import ModelSettings, PreparationSettings, TrainingSettings
from .training import SCORES, run_fold

PROTOCOLS = ("loso", "within")
PAPER = "EEGNet (Lawhern et al. 2018, J. Neural Eng. 15, 056013)"
SCORE_DESCRIPTIONS = dict(
    validation_selected="The paper's protocol: the weights with the lowest validation loss over {epochs} epochs; "
                        "the validation set comes from the training side of the fold. No test label is used.",
    final_epoch="The score after epoch {epochs}, the protocol of version 27. No test label is used.",
    highest_test_epoch="The epoch with the highest test accuracy, chosen with the test labels; biased upward and "
                       "recorded only for comparison with work that reports this way.")


def validation_subjects(others, held_out: int, training: TrainingSettings):
    """The validation subjects of one fold: a random share of the other subjects, fixed by the seed."""
    count = max(1, int(round(training.validation_fraction * len(others))))
    rng = np.random.default_rng([training.seed, held_out])
    return tuple(sorted(int(s) for s in rng.choice(others, count, replace=False)))


def save_model(state, path: Path, dataset: str, channels: int, samples: int, dropout: float,
               model_settings: ModelSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, dataset=dataset, channels=channels, samples=samples, dropout=dropout,
                    model_settings=asdict(model_settings)), path)


def pooled(folds):
    """Scores of a subject over its folds: every sample predicted by the fold that tested it."""
    values = lambda key: [value for fold in folds for value in fold[key]]
    labels = values("labels")
    result = dict(labels=labels)
    for name in SCORES:
        result[name] = classification_metrics(labels, values(f"{name}_predictions"), values(f"{name}_scores"))
        result[f"{name}_predictions"], result[f"{name}_scores"] = values(f"{name}_predictions"), values(f"{name}_scores")
    return result


def run(data_dir, dataset: str, protocol: str = "loso", subjects=None, device=None,
        training: TrainingSettings = TrainingSettings(), model_settings: ModelSettings = ModelSettings(),
        preparation: PreparationSettings = PreparationSettings(), show_progress: bool = True,
        models_dir: Path | None = None) -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    cohort = SUBJECTS[dataset]
    if subjects and not set(subjects) <= set(cohort):
        raise ValueError(f"Subjects must be among the {dataset} subjects of version 27: {list(cohort)}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dropout = model_settings.dropout_cross_subject if protocol == "loso" else model_settings.dropout_within_subject
    data = {}
    for subject in (cohort if protocol == "loso" else subjects or cohort):
        print(f"S{subject}: preparing samples", flush=True)
        data[subject] = load_subject(data_dir, dataset, subject, preparation)
    results = {}
    for subject in subjects or cohort:
        x, y, groups = data[subject]
        if protocol == "loso":
            others = [s for s in cohort if s != subject]
            validation = validation_subjects(others, subject, training)
            train_subjects = [s for s in others if s not in validation]
            join = lambda chosen, i: np.concatenate([data[s][i] for s in chosen])
            record, state = run_fold((join(train_subjects, 0), join(train_subjects, 1)),
                                     (join(validation, 0), join(validation, 1)), (x, y), dropout, device, training,
                                     model_settings, f"Held-out S{subject}", show_progress)
            record.update(training_subjects=train_subjects, validation_subjects=list(validation))
            if models_dir is not None:
                save_model(state, models_dir / f"S{subject}.pt", dataset, x.shape[1], x.shape[2], dropout,
                           model_settings)
            results[str(subject)] = record
        else:
            folds = fold_indices(groups, y)
            records = []
            for k, (_, test_index) in enumerate(folds):
                val_index = folds[(k + 1) % len(folds)][1]
                train_index = np.setdiff1d(np.arange(len(y)), np.concatenate([test_index, val_index]))
                record, state = run_fold((x[train_index], y[train_index]), (x[val_index], y[val_index]),
                                         (x[test_index], y[test_index]), dropout, device, training, model_settings,
                                         f"S{subject} fold {k + 1}/{len(folds)}", show_progress)
                records.append(dict(fold=k + 1, test_index=test_index.tolist(), **record))
                if models_dir is not None:
                    save_model(state, models_dir / f"S{subject}_fold{k + 1}.pt", dataset, x.shape[1], x.shape[2],
                               dropout, model_settings)
            results[str(subject)] = dict(pooled(records), folds=records)
        fold = results[str(subject)]
        minutes = sum(f["training_seconds"] for f in fold.get("folds", [fold])) / 60
        print(f"S{subject} ({minutes:.1f} min): validation-selected {format_metrics(fold['validation_selected'])}\n"
              f"     final epoch          {format_metrics(fold['final_epoch'])}\n"
              f"     highest test epoch   {format_metrics(fold['highest_test_epoch'])}", flush=True)
    summary = {name: summarize({s: r[name] for s, r in results.items()}) for name in SCORES}
    print(f"\nMean over {len(results)} subjects")
    for name in SCORES:
        print(f"  {name.replace('_', ' '):<20} {format_metrics(summary[name])}", flush=True)

    folds = [f for r in results.values() for f in r.get("folds", [r])]
    seconds = [f["training_seconds"] for f in folds]
    memory = [f["peak_gpu_memory_bytes"] for f in folds if f["peak_gpu_memory_bytes"] is not None]
    cost = dict(training_seconds_total=sum(seconds), training_seconds_per_fold_mean=float(np.mean(seconds)),
                training_seconds_per_fold_min=min(seconds), training_seconds_per_fold_max=max(seconds),
                epochs_per_fold=training.epochs, peak_gpu_memory_bytes=max(memory) if memory else None,
                note=f"All {training.epochs} epochs of a fold, including the evaluation of the validation and test "
                     f"samples after every epoch. Data preparation is not included.")
    sample_x = data[next(iter(data))][0]
    complexity = model_complexity(lambda: build_model(sample_x.shape[1], sample_x.shape[2], dropout,
                                                      settings=model_settings), sample_x, device, training.batch_size)
    return dict(paper=PAPER, model="EEGNet-8,2", dataset=dataset, class_names=list(CLASS_NAMES[dataset]),
                protocol="leave_one_subject_out" if protocol == "loso" else "within_subject_cross_validation",
                subjects=list(results), samples="version 27's samples, subjects and folds (datascripts/version27.py)",
                scores=list(SCORES), main_score="validation_selected",
                evaluation={name: text.format(epochs=training.epochs) for name, text in SCORE_DESCRIPTIONS.items()},
                dropout=dropout, summary=summary, results=results, preparation=asdict(preparation),
                model_settings=asdict(model_settings), training=asdict(training), complexity=complexity, cost=cost,
                **environment(device))
