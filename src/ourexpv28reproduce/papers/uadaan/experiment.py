"""UA-DAAN on version 27's samples, with both of version 27's protocols.

UA-DAAN is transductive: the test samples, without their labels, are the
target domain during training (as in the paper, where the test subjects of a
subject-independent fold are the target domain).

  Leave one subject out: source = all other subjects, target = the held-out subject.
  Within subject: source = version 27's nine training folds, target = the tenth fold.

The paper used a validation tenth of the training subjects for a grid search;
the reference implementation trains with fixed settings and scores the last
epoch, which is done here.
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ...datascripts.version27 import CLASS_NAMES, SUBJECTS, fold_indices
from ...evaluation.complexity import environment, model_complexity
from ...evaluation.metrics import classification_metrics, format_metrics, summarize
from .features import load_subject
from .network import build_model
from .settings import FeatureSettings, ModelSettings, TrainingSettings
from .training import SCORES, run_fold

PROTOCOLS = ("loso", "within")
PAPER = "UA-DAAN (Shen et al. 2025, IEEE Trans. Affective Computing 16(3), 2130)"
SCORE_DESCRIPTIONS = dict(
    final_epoch="The score after epoch {epochs}, as the reference implementation scores it, and the protocol of "
                "version 27. The test samples' features (not their labels) are used in training as the target "
                "domain.",
    highest_test_epoch="The epoch with the highest test accuracy, chosen with the test labels; biased upward and "
                       "recorded only for comparison.")


class Predictor(nn.Module):
    """For the cost measurement: one prediction (10 Monte Carlo passes) of the trained network."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.predict(x)


def save_model(state, path: Path, dataset: str, inputs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, dataset=dataset, inputs=inputs), path)


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
        features: FeatureSettings = FeatureSettings(), show_progress: bool = True,
        models_dir: Path | None = None) -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    cohort = SUBJECTS[dataset]
    if subjects and not set(subjects) <= set(cohort):
        raise ValueError(f"Subjects must be among the {dataset} subjects of version 27: {list(cohort)}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {}
    for subject in (cohort if protocol == "loso" else subjects or cohort):
        print(f"S{subject}: computing differential entropy", flush=True)
        data[subject] = load_subject(data_dir, dataset, subject, features)
    results = {}
    for subject in subjects or cohort:
        x, y, groups = data[subject]
        if protocol == "loso":
            others = [s for s in cohort if s != subject]
            source = (np.concatenate([data[s][0] for s in others]), np.concatenate([data[s][1] for s in others]))
            record, state = run_fold(source, (x, y), device, training, model_settings, f"Held-out S{subject}",
                                     show_progress)
            if models_dir is not None:
                save_model(state, models_dir / f"S{subject}.pt", dataset, x.shape[1])
            results[str(subject)] = record
        else:
            records = []
            for k, (train_index, test_index) in enumerate(fold_indices(groups, y)):
                record, state = run_fold((x[train_index], y[train_index]), (x[test_index], y[test_index]), device,
                                         training, model_settings, f"S{subject} fold {k + 1}/10", show_progress)
                records.append(dict(fold=k + 1, test_index=test_index.tolist(), **record))
                if models_dir is not None:
                    save_model(state, models_dir / f"S{subject}_fold{k + 1}.pt", dataset, x.shape[1])
            results[str(subject)] = dict(pooled(records), folds=records)
        result = results[str(subject)]
        minutes = sum(f["training_seconds"] for f in result.get("folds", [result])) / 60
        print(f"S{subject} ({minutes:.1f} min): final epoch {format_metrics(result['final_epoch'])}\n"
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
                note=f"All {training.epochs} epochs of a fold, including scoring the test and training samples after "
                     f"every epoch. Feature extraction (differential entropy) is not included.")
    sample_x = data[next(iter(data))][0]
    complexity = model_complexity(lambda: Predictor(build_model(sample_x.shape[1], model_settings, training)),
                                  sample_x, device, training.target_batch)
    complexity["note_model"] = ("Parameters include the three domain discriminators, which are used only in training; "
                                "a prediction is 10 Monte Carlo passes of the extractor and the classifier.")
    return dict(paper=PAPER, model="UA-DAAN", dataset=dataset, class_names=list(CLASS_NAMES[dataset]),
                protocol="leave_one_subject_out" if protocol == "loso" else "within_subject_cross_validation",
                subjects=list(results), samples="version 27's samples, subjects and folds (datascripts/version27.py)",
                scores=list(SCORES), main_score="final_epoch",
                evaluation={name: text.format(epochs=training.epochs) for name, text in SCORE_DESCRIPTIONS.items()},
                unlabeled_test_use="Transductive: the test samples' features are the target domain during training.",
                input_features=int(sample_x.shape[1]), summary=summary, results=results, features=asdict(features),
                model_settings=asdict(model_settings), training=asdict(training), complexity=complexity, cost=cost,
                **environment(device))
