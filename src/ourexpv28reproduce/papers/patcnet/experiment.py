"""PA-TCNet on shared comparison samples by default; optional original paper inputs."""
from dataclasses import asdict
import random

import numpy as np
import torch

from ...evaluation.complexity import environment, model_complexity
from ...evaluation.metrics import classification_metrics, format_metrics, summarize
from ...datascripts.version27 import CLASS_NAMES, fold_indices
from .channels import channel_info as roi_channel_info
from .network import build_model
from .preparation import load_comparison_subject, load_subject
from .settings import ModelSettings, TrainingSettings, dataset_settings
from .training import run_fold

PROTOCOLS = ("loso", "within")

COHORTS = ("version27", "published", "distinct")   # liu2024: the paper's 24 subjects, or 23 without the copy 49 of subject 47


def seed_everything(seed: int) -> None:
    """Once, before the first fold, as pytorch_lightning.seed_everything in the published code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subjects_of(dataset: str, cohort: str) -> tuple[int, ...]:
    return dataset_settings(dataset, cohort).subjects


def run(data_dir, dataset: str, cohort: str = "version27", held_out=None, device=None,
        training: TrainingSettings = TrainingSettings(), model_settings: ModelSettings = ModelSettings(),
        show_progress: bool = True, protocol: str = "loso") -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    if training.epochs < 1 or training.batch_size < 2 or training.patience < 1:
        raise ValueError("epochs and patience must be positive; batch_size must be at least 2")
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}")
    settings = dataset_settings(dataset, cohort)
    if held_out and not set(held_out) <= set(settings.subjects):
        raise ValueError(f"Held-out subjects must be in the {cohort} cohort of {dataset}: {list(settings.subjects)}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = roi_channel_info(dataset, cohort) if dataset in ("liu2024", "bci2a") else None
    seed_everything(training.seed)
    data = {}
    for subject in (settings.subjects if protocol == "loso" else held_out or settings.subjects):
        print(f"S{subject}: preparing trials", flush=True)
        if cohort == "version27":
            data[subject] = load_comparison_subject(data_dir, dataset, subject)
        else:
            x, y = load_subject(data_dir, dataset, subject)
            data[subject] = x, y, np.arange(len(y))
    results = {}
    for test_subject in held_out or settings.subjects:
        if protocol == "loso":
            others = [s for s in settings.subjects if s != test_subject]
            x_source = np.concatenate([data[s][0] for s in others])
            y_source = np.concatenate([data[s][1] for s in others])
            fold = run_fold(x_source, y_source, *data[test_subject][:2], info, settings, device, model_settings, training,
                            f"Held-out S{test_subject}", show_progress)
        else:
            x, y, groups = data[test_subject]
            records = []
            # Keep imagery/rest samples of one Liu trial together, exactly as version 27.
            for k, (train, test) in enumerate(fold_indices(groups, y)):
                record = run_fold(x[train], y[train], x[test], y[test], info, settings, device,
                                  model_settings, training, f"S{test_subject} fold {k + 1}/10", show_progress)
                records.append(dict(fold=k + 1, train_index=train.tolist(), test_index=test.tolist(), **record))
            fold = dict(folds=records, labels=[v for r in records for v in r["labels"]])
            for name in ("final_epoch", "as_published"):
                for suffix in ("predictions", "scores"):
                    key = f"{name}_{suffix}"
                    fold[key] = [v for r in records for v in r[key]]
                fold[name] = classification_metrics(fold["labels"], fold[f"{name}_predictions"], fold[f"{name}_scores"])
        fold["sample_groups"] = data[test_subject][2].tolist()
        results[str(test_subject)] = fold
        print(f"S{test_subject}: final epoch {format_metrics(fold['final_epoch'])}\n"
              f"      test-selected {format_metrics(fold['as_published'])}", flush=True)
    summary = dict(final_epoch=summarize({k: v["final_epoch"] for k, v in results.items()}),
                   as_published=summarize({k: v["as_published"] for k, v in results.items()}))
    print(f"\nMean over {len(results)} held-out subjects\n"
          f"  final epoch (fixed {training.epochs} epochs, chosen without test labels): {format_metrics(summary['final_epoch'])}\n"
          f"  as published (best test epoch up to early stopping on test loss):    {format_metrics(summary['as_published'])}",
          flush=True)
    folds = [f for r in results.values() for f in r.get("folds", [r])]
    seconds = [fold["training_seconds"] for fold in folds]
    memory = [fold["peak_gpu_memory_bytes"] for fold in folds if fold["peak_gpu_memory_bytes"] is not None]
    cost = dict(training_seconds_total=sum(seconds), training_seconds_per_fold_mean=sum(seconds) / len(seconds),
                training_seconds_per_fold_min=min(seconds), training_seconds_per_fold_max=max(seconds),
                epochs_per_fold=training.epochs, peak_gpu_memory_bytes=max(memory) if memory else None,
                note=f"All {training.epochs} epochs of a fold, including the evaluation after every epoch; the "
                     f"published code stops at as_published.stopping_epoch, which can be earlier. "
                     f"Data preparation is not included.")
    complexity = model_complexity(lambda: build_model(dataset, model_settings, settings=settings), data[next(iter(data))][0],
                                  device, training.batch_size)
    return dict(paper="PA-TCNet (Wang et al. 2026, arXiv:2604.16554)", dataset=dataset, paper_dataset=settings.name,
                cohort=cohort, subjects=list(results), cohort_subjects=list(settings.subjects), protocol="leave_one_subject_out" if protocol == "loso" else "within_subject_cross_validation",
                methodology=("PATCNet evaluated under our common experimental protocol" if cohort == "version27"
                             else "paper LOSO reproduction" if protocol == "loso" and dataset != "bci2a"
                             else "extension of paper methodology"),
                class_names=list(CLASS_NAMES[dataset]) if cohort == "version27" else ["left hand", "right hand"],
                scores=["final_epoch", "as_published"], main_score="final_epoch",
                adaptation="Transductive: unlabeled test inputs are used by PGTC during training.",
                samples=("version 27's samples, subjects and folds (datascripts/version27.py)" if cohort == "version27"
                         else "paper stroke windows/cohorts; BCI2a extension with version 27 samples"),
                pgtc="ROI templates and confidence" if info else "confidence only (no channel names)",
                evaluation=dict(
                    final_epoch=f"Score after epoch {training.epochs}; no test label is used to obtain it.",
                    as_published="Test-selected diagnostic (paper selection rule; extension for BCI2a/within): the epoch with the highest accuracy on the held-out "
                                 "subject, up to the epoch at which early stopping on the held-out subject's loss "
                                 "(configured patience) would end training. Chosen with the test labels."),
                summary=summary, results=results, settings=asdict(settings), model=asdict(model_settings),
                training=asdict(training), complexity=complexity, cost=cost, **environment(device))
