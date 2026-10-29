"""SlimSeiz on version 27's samples, with both of version 27's protocols.

For every test fold: select k channels on the training side, split 20 % of
the training side off as the validation set, train on the rest, and test.

  Within subject: version 27's 10 folds (the paper also uses 10 folds); the
  validation set is 20 % of the training trials, chosen at random (trials
  kept together), as the code takes 20 % of the training indices.
  Leave one subject out: the paper is patient-specific; the validation set is
  20 % of the training subjects, so that validation is across subjects, as the
  test is.

The channel selection uses the training side including its validation part
(labels of the training side only).
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ...datascripts.version27 import CLASS_NAMES, SUBJECTS, fold_indices
from ...evaluation.complexity import environment, model_complexity
from ...evaluation.metrics import classification_metrics, format_metrics, summarize
from .channels import select_channels
from .network import build_model
from .preparation import load_subject
from .settings import ChannelSelectionSettings, TrainingSettings
from .training import SCORES, run_fold

PROTOCOLS = ("loso", "within")
PAPER = "SlimSeiz (Lu et al. 2025, IEEE ISCAS, DOI 10.1109/ISCAS56072.2025.11043364)"
SCORE_DESCRIPTIONS = dict(
    validation_selected="The code's protocol: the epoch with the highest validation accuracy over {epochs} epochs; "
                        "the validation set comes from the training side of the fold. No test label is used.",
    final_epoch="The score after epoch {epochs}, the protocol of version 27. No test label is used.",
    highest_test_epoch="The epoch with the highest test accuracy, chosen with the test labels; biased upward and "
                       "recorded only for comparison.")


def validation_part(groups, share: float, seed: int, key: int):
    """A random share of the distinct `groups` (trials or subjects), at least one."""
    unique = np.unique(groups)
    rng = np.random.default_rng([seed, key])
    return np.isin(groups, rng.choice(unique, max(1, int(round(share * len(unique)))), replace=False))


def save_model(state, path: Path, dataset: str, channels, samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, dataset=dataset, selected_channels=list(channels), samples=samples), path)


def pooled(folds):
    values = lambda key: [value for fold in folds for value in fold[key]]
    labels = values("labels")
    result = dict(labels=labels)
    for name in SCORES:
        result[name] = classification_metrics(labels, values(f"{name}_predictions"), values(f"{name}_scores"))
        result[f"{name}_predictions"], result[f"{name}_scores"] = values(f"{name}_predictions"), values(f"{name}_scores")
    return result


def fold(train_x, train_y, train_groups, test, key, dataset, training, selection, device, description,
         show_progress):
    """Select channels, split off the validation set, train and test one fold."""
    chosen, counts = select_channels(train_x, train_y, selection, seed=training.seed + key)
    held = validation_part(train_groups, training.validation_share, training.seed, key)
    pick = lambda a: a[:, chosen]
    record, state = run_fold((pick(train_x[~held]), train_y[~held]), (pick(train_x[held]), train_y[held]),
                             (pick(test[0]), test[1]), device, training, description, show_progress)
    record.update(selected_channels=chosen, channel_counts=counts)
    return record, state, chosen


def run(data_dir, dataset: str, protocol: str = "loso", subjects=None, device=None,
        training: TrainingSettings = TrainingSettings(), selection: ChannelSelectionSettings = ChannelSelectionSettings(),
        show_progress: bool = True, models_dir: Path | None = None) -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    cohort = SUBJECTS[dataset]
    if subjects and not set(subjects) <= set(cohort):
        raise ValueError(f"Subjects must be among the {dataset} subjects of version 27: {list(cohort)}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            record, state, chosen = fold(train_x, train_y, owner, (x, y), subject, dataset, training, selection, device,
                                         f"Held-out S{subject}", show_progress)
            record["validation_subjects"] = sorted(set(owner[validation_part(owner, training.validation_share,
                                                                              training.seed, subject)].tolist()))
            if models_dir is not None:
                save_model(state, models_dir / f"S{subject}.pt", dataset, chosen, x.shape[2])
            results[str(subject)] = record
        else:
            records = []
            for k, (train_index, test_index) in enumerate(fold_indices(groups, y)):
                record, state, chosen = fold(x[train_index], y[train_index], groups[train_index],
                                             (x[test_index], y[test_index]), 1000 * subject + k, dataset, training,
                                             selection, device, f"S{subject} fold {k + 1}/10", show_progress)
                records.append(dict(fold=k + 1, test_index=test_index.tolist(), **record))
                if models_dir is not None:
                    save_model(state, models_dir / f"S{subject}_fold{k + 1}.pt", dataset, chosen, x.shape[2])
            results[str(subject)] = dict(pooled(records), folds=records)
        result = results[str(subject)]
        minutes = sum(f["training_seconds"] for f in result.get("folds", [result])) / 60
        print(f"S{subject} ({minutes:.1f} min): validation-selected {format_metrics(result['validation_selected'])}\n"
              f"     final epoch          {format_metrics(result['final_epoch'])}\n"
              f"     highest test epoch   {format_metrics(result['highest_test_epoch'])}", flush=True)
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
                     f"samples after every epoch. Data preparation and channel selection are not included.")
    k = min(selection.channels, data[next(iter(data))][0].shape[1])
    sample_x = data[next(iter(data))][0][:, :k]
    complexity = model_complexity(lambda: build_model(k), sample_x, device, training.batch_size)
    complexity["note_outputs"] = "The model returns (logits, features); FLOPs and latency are for the whole forward pass."
    return dict(paper=PAPER, model="SlimSeiz", dataset=dataset, class_names=list(CLASS_NAMES[dataset]),
                protocol="leave_one_subject_out" if protocol == "loso" else "within_subject_cross_validation",
                subjects=list(results), samples="version 27's samples, subjects and folds (datascripts/version27.py)",
                scores=list(SCORES), main_score="validation_selected",
                evaluation={name: text.format(epochs=training.epochs) for name, text in SCORE_DESCRIPTIONS.items()},
                selected_channels_count=k, summary=summary, results=results, training=asdict(training),
                channel_selection=asdict(selection), complexity=complexity, cost=cost, **environment(device))
