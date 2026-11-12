"""Leave one subject out: train on the other subjects, test once on the held-out one."""
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
import time

import torch
from tqdm import tqdm

from ..datascripts import DataConfig, build_subject_graphs, liu2024_config
from ..datascripts.roi import CHANNEL_NAMES
from ..model import ModelConfig
from ..model.config import PATHS
from .config import TrainConfig
from .complexity import embeddings, model_complexity, synchronize
from .engine import ensemble_scores
from .metrics import classification_metrics, format_metrics, print_summary, summarize
from .validation import select_epochs, split_subjects

PACKAGES = ("torch", "torch-geometric", "numpy", "scipy", "scikit-learn", "mne")


def select_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(preference)


def save_model(model, path, seed, model_config, data_config, config, selection=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, seed=seed,
                    model_config=asdict(model_config), data_config=asdict(data_config),
                    train_config=asdict(config), validation=selection), path)


def run_fold(training, testing, name, data_config, model_config, config, device, save_to=None, selection=None):
    """Fresh final refit; test labels are read only after all final predictions exist."""
    from .validation import Member
    synchronize(device)
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logits, histories, calibration_states = [], [], []
    for member_index in range(config.ensemble):
        member = Member(training, testing, model_config, config, config.seed + member_index, device)
        for epoch in tqdm(range(1, config.epochs + 1), desc=f"{name}, refit {member_index + 1}/{config.ensemble}",
                          disable=not config.show_progress):
            final = member.epoch(epoch, score_target=epoch == config.epochs)
        logits.append(final)
        histories.append(member.history)
        if member_index == 0:
            features = embeddings(member.model, member.batches, device)
        if member.calibration is not None:
            calibration_states.append(dict(labelled_fraction=float((member.calibration.targets.sum(1) > 0).float().mean())))
        if save_to is not None:
            save_model(member.model, save_to.with_name(f"{save_to.name}_model{member_index + 1}.pt"),
                       config.seed + member_index, model_config, data_config, config, selection)
        del member
    predictions, scores = ensemble_scores(torch.stack(logits), config.center_logits)
    labels = [int(graph.y) for graph in testing]
    synchronize(device)
    result = dict(classification_metrics(labels, predictions.tolist(), scores.tolist()),
                  labels=labels, predictions=predictions.tolist(), scores=scores.tolist(),
                  selected_epoch=config.epochs, refit_epochs=config.epochs,
                  training_samples=len(training), testing_samples=len(testing), history=histories,
                  training_seconds=time.perf_counter() - start,
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  embeddings=features)
    if selection is not None:
        result['validation'] = selection
        result['refit_training_seconds'] = result['training_seconds']
        result['training_seconds'] += selection['training_seconds']
    if calibration_states:
        result['calibration'] = calibration_states
    return result


def run_information(data_config: DataConfig, model_config: ModelConfig, config: TrainConfig, device,
                    test_graphs) -> dict:
    """What every result file records about the run, including the size and cost of the model."""
    complexity = model_complexity(model_config, test_graphs, device, config.batch_size)
    complexity["models_per_decision"] = config.ensemble
    return dict(dataset=data_config.dataset, task=data_config.task, class_names=list(data_config.class_names),
                data_config=asdict(data_config), model_config=asdict(model_config), train_config=asdict(config),
                device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                packages={name: version(name) for name in PACKAGES}, complexity=complexity)


def run_cost(results: dict) -> dict:
    """Training time and peak memory over the folds."""
    seconds = [fold["training_seconds"] for fold in results]
    memory = [fold["peak_gpu_memory_bytes"] for fold in results if fold["peak_gpu_memory_bytes"] is not None]
    return dict(training_seconds_total=sum(seconds), training_seconds_per_fold_mean=sum(seconds) / len(seconds),
                training_seconds_per_fold_min=min(seconds), training_seconds_per_fold_max=max(seconds),
                peak_gpu_memory_bytes=max(memory) if memory else None,
                note="Validation selection plus fresh final refit of all models; graph building is excluded.")


def run_loso(data_dir, data_config: DataConfig | None = None, config: TrainConfig = TrainConfig(),
             held_out_subjects=None, models_dir: Path | None = None, paths=PATHS) -> dict:
    """Every subject in `held_out_subjects` (default: all) is the test subject of one fold.

    `models_dir`: save the models of every fold there, as `S<subject>_model<k>.pt`.
    `paths`: the paths of the model; fewer than all three is an ablation.
    """
    data_config = data_config or liu2024_config()
    if config.target_calibration and data_config.dataset not in CHANNEL_NAMES:
        print(f"Target calibration is switched off: {data_config.dataset} has no channel names", flush=True)
        config = replace(config, target_calibration=False)
    config.validate()
    model_config = ModelConfig.for_data(data_config, paths=tuple(paths))
    device = select_device(config.device)
    held_out_subjects = tuple(held_out_subjects or data_config.subjects)
    if not set(held_out_subjects) <= set(data_config.subjects):
        raise ValueError("Held-out subjects must be among the configured subjects")
    graphs = {}
    for subject in data_config.subjects:
        print(f"S{subject}: building graphs", flush=True)
        graphs[subject] = build_subject_graphs(data_dir, subject, data_config)
    results = {}
    for held_out in held_out_subjects:
        train_subjects, validation_subjects = split_subjects(data_config.subjects, held_out, config.seed,
                                                            config.validation_fraction)
        selection_training = [g for subject in train_subjects for g in graphs[subject]]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        selection = select_epochs(selection_training, {s: graphs[s] for s in validation_subjects},
                                  model_config, config, device, f"S{held_out} validation")
        selection_peak = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None
        selection['split'] = dict(training_subjects=train_subjects, validation_subjects=validation_subjects,
                                  test_subjects=[held_out], seed=config.seed, held_out_subject=held_out)
        selection['sample_assignments'] = {
            str(subject): [dict(trial_id=int(g.trial_id), sample_id=int(g.sample_id)) for g in graphs[subject]]
            for subject in train_subjects + validation_subjects}
        training = [g for subject in data_config.subjects if subject != held_out for g in graphs[subject]]
        final_config = replace(config, epochs=selection['selected_epoch'])
        result = run_fold(training, graphs[held_out], f"Held-out S{held_out}", data_config,
                          model_config, final_config, device,
                          models_dir / f"S{held_out}" if models_dir else None, selection)
        if selection_peak is not None:
            result['peak_gpu_memory_bytes'] = max(result['peak_gpu_memory_bytes'], selection_peak)
        result['refit_subjects'] = [s for s in data_config.subjects if s != held_out]
        results[str(held_out)] = result
        print(f"S{held_out}: selected epoch {selection['selected_epoch']} "
              f"(validation stopped at {selection['stopped_epoch']}), {format_metrics(result)}", flush=True)
    summary = summarize(results)
    print_summary(f"Mean over {len(results)} held-out subjects", summary)
    features = {subject: fold.pop("embeddings") for subject, fold in results.items()}
    return dict(protocol="leave_one_subject_out",
                evaluation="Validation-selected epoch count, followed by fresh refit on all non-test subjects. "
                           "Test labels are used only for the final evaluation; no highest-test-epoch selection.",
                summary=summary, subjects=results,
                unlabeled_test_subject_use=dict(
                    alignment="Euclidean alignment uses the held-out subject's own samples",
                    center_logits=config.center_logits, target_calibration=config.target_calibration,
                    note="All use the held-out subject's samples without their labels."),
                cost=run_cost(list(results.values())), embeddings=features,
                **run_information(data_config, model_config, config, device, graphs[held_out_subjects[0]]))
