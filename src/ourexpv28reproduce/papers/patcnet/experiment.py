"""PA-TCNet leave one subject out, over a dataset's cohort, as in train_loso of the published code."""
from dataclasses import asdict, replace
import random

import numpy as np
import torch

from ...evaluation.complexity import environment, model_complexity
from ...evaluation.metrics import format_metrics, summarize
from .channels import channel_info as roi_channel_info
from .network import build_model
from .preparation import load_subject
from .settings import DATASETS, ModelSettings, TrainingSettings
from .training import run_fold

COHORTS = ("published", "distinct")   # liu2024: the paper's 24 subjects, or 23 without the copy 49 of subject 47


def seed_everything(seed: int) -> None:
    """Once, before the first fold, as pytorch_lightning.seed_everything in the published code."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subjects_of(dataset: str, cohort: str) -> tuple[int, ...]:
    subjects = DATASETS[dataset].subjects
    if cohort == "distinct" and dataset == "liu2024":
        subjects = tuple(s for s in subjects if s != 49)
    return subjects


def run(data_dir, dataset: str, cohort: str = "published", held_out=None, device=None,
        training: TrainingSettings = TrainingSettings(), model_settings: ModelSettings = ModelSettings(),
        show_progress: bool = True) -> dict:
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}")
    settings = replace(DATASETS[dataset], subjects=subjects_of(dataset, cohort))
    if held_out and not set(held_out) <= set(settings.subjects):
        raise ValueError(f"Held-out subjects must be in the {cohort} cohort of {dataset}: {list(settings.subjects)}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = roi_channel_info(dataset) if dataset == "liu2024" else None
    seed_everything(training.seed)
    data = {}
    for subject in settings.subjects:
        print(f"S{subject}: preparing trials", flush=True)
        data[subject] = load_subject(data_dir, dataset, subject)
    results = {}
    for test_subject in held_out or settings.subjects:
        others = [s for s in settings.subjects if s != test_subject]
        x_source = np.concatenate([data[s][0] for s in others])
        y_source = np.concatenate([data[s][1] for s in others])
        fold = run_fold(x_source, y_source, *data[test_subject], info, settings, device, model_settings, training,
                        f"Held-out S{test_subject}", show_progress)
        results[str(test_subject)] = fold
        print(f"S{test_subject}: final epoch  {format_metrics(fold['final_epoch'])}\n"
              f"      as published (epoch {fold['as_published']['epoch']}, stop {fold['as_published']['stopping_epoch']}): "
              f"{format_metrics(fold['as_published'])}", flush=True)
    summary = dict(final_epoch=summarize({k: v["final_epoch"] for k, v in results.items()}),
                   as_published=summarize({k: v["as_published"] for k, v in results.items()}))
    print(f"\nMean over {len(results)} held-out subjects\n"
          f"  final epoch (fixed {training.epochs} epochs, chosen without test labels): {format_metrics(summary['final_epoch'])}\n"
          f"  as published (best test epoch up to early stopping on test loss):    {format_metrics(summary['as_published'])}",
          flush=True)
    seconds = [fold["training_seconds"] for fold in results.values()]
    memory = [fold["peak_gpu_memory_bytes"] for fold in results.values() if fold["peak_gpu_memory_bytes"] is not None]
    cost = dict(training_seconds_total=sum(seconds), training_seconds_per_fold_mean=sum(seconds) / len(seconds),
                training_seconds_per_fold_min=min(seconds), training_seconds_per_fold_max=max(seconds),
                epochs_per_fold=training.epochs, peak_gpu_memory_bytes=max(memory) if memory else None,
                note=f"All {training.epochs} epochs of a fold, including the evaluation after every epoch; the "
                     f"published code stops at as_published.stopping_epoch, which can be earlier. "
                     f"Data preparation is not included.")
    complexity = model_complexity(lambda: build_model(dataset, model_settings), data[settings.subjects[0]][0],
                                  device, training.batch_size)
    return dict(paper="PA-TCNet (Wang et al. 2026, arXiv:2604.16554)", dataset=dataset, paper_dataset=settings.name,
                cohort=cohort, subjects=list(settings.subjects), protocol="leave_one_subject_out",
                pgtc="ROI templates and confidence" if info else "confidence only (no channel names)",
                evaluation=dict(
                    final_epoch=f"Score after epoch {training.epochs}; no test label is used to obtain it.",
                    as_published="The published protocol: the epoch with the highest accuracy on the held-out "
                                 "subject, up to the epoch at which early stopping on the held-out subject's loss "
                                 "(patience 30) would end training. Chosen with the test labels."),
                summary=summary, results=results, settings=asdict(settings), model=asdict(model_settings),
                training=asdict(training), complexity=complexity, cost=cost, **environment(device))
