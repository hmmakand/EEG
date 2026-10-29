"""EEG Conformer on version 27's samples, with both of version 27's protocols.

The paper evaluates within subject (BCI IV 2a: session 1 for training,
session 2 for testing). Here:

  Leave one subject out: train on all other subjects, test on the held-out one.
  Within subject: version 27's 10 folds; train on nine, test on the tenth.

There is no validation set, as in the code. The z-score uses the fold's
training samples.
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
PAPER = "EEG Conformer (Song et al. 2023, IEEE TNSRE 31, 710)"
SCORE_DESCRIPTIONS = dict(
    final_epoch="The score after epoch {epochs}, the protocol of version 27. No test label is used.",
    highest_test_epoch="The published code's score: the epoch with the highest test accuracy over {epochs} "
                       "epochs, chosen with the test labels. Biased upward; for comparison with the paper.")


def save_model(state, path: Path, dataset: str, channels: int, samples: int, model_settings: ModelSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, dataset=dataset, channels=channels, samples=samples,
                    model_settings=asdict(model_settings)), path)


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
    data = {}
    for subject in (cohort if protocol == "loso" else subjects or cohort):
        print(f"S{subject}: preparing samples", flush=True)
        data[subject] = load_subject(data_dir, dataset, subject, preparation)
    results = {}
    for subject in subjects or cohort:
        x, y, groups = data[subject]
        if protocol == "loso":
            others = [s for s in cohort if s != subject]
            train = (np.concatenate([data[s][0] for s in others]), np.concatenate([data[s][1] for s in others]))
            record, state = run_fold(train, (x, y), device, training, model_settings, f"Held-out S{subject}",
                                     show_progress)
            if models_dir is not None:
                save_model(state, models_dir / f"S{subject}.pt", dataset, x.shape[1], x.shape[2], model_settings)
            results[str(subject)] = record
        else:
            records = []
            for k, (train_index, test_index) in enumerate(fold_indices(groups, y)):
                record, state = run_fold((x[train_index], y[train_index]), (x[test_index], y[test_index]), device,
                                         training, model_settings, f"S{subject} fold {k + 1}/10", show_progress)
                records.append(dict(fold=k + 1, test_index=test_index.tolist(), **record))
                if models_dir is not None:
                    save_model(state, models_dir / f"S{subject}_fold{k + 1}.pt", dataset, x.shape[1], x.shape[2],
                               model_settings)
            results[str(subject)] = dict(pooled(records), folds=records)
        fold = results[str(subject)]
        minutes = sum(f["training_seconds"] for f in fold.get("folds", [fold])) / 60
        print(f"S{subject} ({minutes:.1f} min): final epoch {format_metrics(fold['final_epoch'])}\n"
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
                note=f"All {training.epochs} epochs of a fold, including the evaluation of the test samples after "
                     f"every epoch. Data preparation is not included.")
    sample_x = data[next(iter(data))][0]
    complexity = model_complexity(lambda: build_model(sample_x.shape[1], sample_x.shape[2], settings=model_settings),
                                  sample_x, device, training.batch_size)
    return dict(paper=PAPER, model="EEG Conformer", dataset=dataset, class_names=list(CLASS_NAMES[dataset]),
                protocol="leave_one_subject_out" if protocol == "loso" else "within_subject_cross_validation",
                subjects=list(results), samples="version 27's samples, subjects and folds (datascripts/version27.py)",
                scores=list(SCORES), main_score="final_epoch",
                evaluation={name: text.format(epochs=training.epochs) for name, text in SCORE_DESCRIPTIONS.items()},
                summary=summary, results=results, preparation=asdict(preparation),
                model_settings=asdict(model_settings), training=asdict(training), complexity=complexity, cost=cost,
                **environment(device))
