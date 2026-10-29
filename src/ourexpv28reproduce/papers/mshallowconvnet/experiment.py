"""M-ShallowConvNet on version 27's samples, with both of version 27's protocols.

The paper evaluates within subject only: five-fold cross-validation on the
training session gives five models, which are bagged on the test session.
Here the test samples are version 27's, and the five-fold split is made on
the training side of each fold:

  Leave one subject out: the other subjects are split into five groups of
  subjects (shuffled, seeded); model m validates on group m and trains on the
  rest. Splitting by subject keeps the validation cross-subject, as the test is.

  Within subject: for each of version 27's 10 test folds, the other nine folds
  are split into five parts (trials kept together, shuffled, seeded); model m
  validates on part m and trains on the rest.
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ...datascripts.splits import split_groups
from ...datascripts.version27 import CLASS_NAMES, SUBJECTS, fold_indices
from ...evaluation.complexity import environment, model_complexity
from ...evaluation.metrics import classification_metrics, format_metrics, summarize
from .network import build_model
from .preparation import load_subject
from .settings import ModelSettings, PreparationSettings, TrainingSettings
from .training import SCORES, run_fold

PROTOCOLS = ("loso", "within")
PAPER = "M-ShallowConvNet (Kim, Lee and Lee 2022, IEEE Access 10, 96984)"
SCORE_DESCRIPTIONS = dict(
    validation_selected="The paper's protocol: five models, each at its lowest validation loss over {epochs} "
                        "epochs, bagged; the validation sets come from the training side of the fold. No test "
                        "label is used.",
    final_epoch="The five models after epoch {epochs}, bagged: the protocol of version 27. No test label is used.",
    highest_test_epoch="The epoch at which the bagged models are most accurate on the test samples, chosen with "
                       "the test labels; biased upward and recorded only for comparison.")


def save_models(states, folder: Path, name: str, dataset: str, channels: int, samples: int,
                model_settings: ModelSettings) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for m, state in enumerate(states, start=1):
        torch.save(dict(state_dict=state, dataset=dataset, channels=channels, samples=samples,
                        model_settings=asdict(model_settings)), folder / f"{name}_model{m}.pt")


def pooled(folds):
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
    torch.backends.cudnn.benchmark = True              # as training.py of the published code
    ratio = training.cutcat_ratio[dataset]
    data = {}
    for subject in (cohort if protocol == "loso" else subjects or cohort):
        print(f"S{subject}: preparing samples", flush=True)
        data[subject] = load_subject(data_dir, dataset, subject, preparation)
    results = {}
    for subject in subjects or cohort:
        x, y, groups = data[subject]
        if protocol == "loso":
            others = [s for s in cohort if s != subject]
            parts = split_groups(others, training.models, training.seed, subject)
            join = lambda chosen, i: np.concatenate([data[s][i] for s in chosen])
            splits = [((join([s for s in others if s not in part], 0), join([s for s in others if s not in part], 1)),
                       (join(part, 0), join(part, 1))) for part in parts]
            record, states = run_fold(splits, x, y, ratio, device, training, model_settings,
                                      f"Held-out S{subject}", show_progress)
            record["validation_subjects"] = parts
            if models_dir is not None:
                save_models(states, models_dir, f"S{subject}", dataset, x.shape[1], x.shape[2], model_settings)
            results[str(subject)] = record
        else:
            records = []
            for k, (train_index, test_index) in enumerate(fold_indices(groups, y)):
                parts = split_groups(groups[train_index], training.models, training.seed, 1000 * subject + k)
                splits = []
                for part in parts:
                    held = np.isin(groups[train_index], part)
                    fit, val = train_index[~held], train_index[held]
                    splits.append(((x[fit], y[fit]), (x[val], y[val])))
                record, states = run_fold(splits, x[test_index], y[test_index], ratio, device, training,
                                          model_settings, f"S{subject} fold {k + 1}/10", show_progress)
                records.append(dict(fold=k + 1, test_index=test_index.tolist(), **record))
                if models_dir is not None:
                    save_models(states, models_dir, f"S{subject}_fold{k + 1}", dataset, x.shape[1], x.shape[2],
                                model_settings)
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
                epochs_per_fold=training.epochs, models_per_fold=training.models,
                peak_gpu_memory_bytes=max(memory) if memory else None,
                note=f"All {training.models} models of a fold, {training.epochs} epochs each, including the "
                     f"evaluation of the validation and test samples after every epoch. Data preparation is not "
                     f"included.")
    sample_x = data[next(iter(data))][0]
    complexity = model_complexity(lambda: build_model(sample_x.shape[1], sample_x.shape[2], settings=model_settings),
                                  sample_x, device, training.batch_size)
    complexity["models_per_decision"] = training.models
    return dict(paper=PAPER, model="M-ShallowConvNet", dataset=dataset, class_names=list(CLASS_NAMES[dataset]),
                protocol="leave_one_subject_out" if protocol == "loso" else "within_subject_cross_validation",
                subjects=list(results), samples="version 27's samples, subjects and folds (datascripts/version27.py)",
                scores=list(SCORES), main_score="validation_selected",
                evaluation={name: text.format(epochs=training.epochs) for name, text in SCORE_DESCRIPTIONS.items()},
                cutcat_ratio=ratio, summary=summary, results=results, preparation=asdict(preparation),
                model_settings=asdict(model_settings), training=asdict(training), complexity=complexity, cost=cost,
                **environment(device))
