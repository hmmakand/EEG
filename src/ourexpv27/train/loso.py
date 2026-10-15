"""Leave one subject out: train on the other subjects, test once on the held-out one."""
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..datascripts import DataConfig, build_subject_graphs, liu2024_config
from ..datascripts.roi import CHANNEL_NAMES
from ..model import EEGGraphModel, ModelConfig
from ..model.config import PATHS
from .calibration import TargetCalibration, train_epoch_calibrated
from .config import TrainConfig
from .complexity import embeddings, model_complexity, synchronize
from .engine import ensemble_scores, test_logits, train_epoch
from .metrics import classification_metrics, format_metrics, print_summary, summarize

PACKAGES = ("torch", "torch-geometric", "numpy", "scipy", "scikit-learn", "mne")


def select_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(preference)


def train_model(loader, test_batches, calibration: TargetCalibration | None, seed: int, model_config: ModelConfig,
                config: TrainConfig, device, description="", labels=None):
    """Train one model for the configured number of epochs; return it, its logits of the test samples
    after every epoch, [epochs, samples, classes], and its training history.

    The history holds, per epoch, the mean loss and the accuracy on the
    training batches (measured while training, with dropout), the share of
    test samples with a calibration target and how many of those targets are
    right (from `labels`, a diagnostic only), and the training time.

    With a `calibration`, the warm-up epochs use the training samples alone
    and the epochs after it also train on the test subject's samples, with the
    model's own checked predictions as targets; the targets are refreshed
    after every epoch. No validation set, no early stopping, no choice of an
    epoch.
    """
    torch.manual_seed(seed)
    model = EEGGraphModel(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    logits = []
    history = dict(train_loss=[], train_accuracy=[], calibration_labelled_fraction=[],
                   calibration_target_accuracy=[])
    synchronize(device)
    start = time.perf_counter()
    for epoch in tqdm(range(config.epochs), desc=description, leave=False, disable=not config.show_progress):
        if calibration is not None and epoch >= config.calibration_warmup_epochs:
            loss, accuracy = train_epoch_calibrated(model, loader, calibration, optimizer, criterion, device)
        else:
            loss, accuracy = train_epoch(model, loader, optimizer, criterion, device)
        history["train_loss"].append(loss)
        history["train_accuracy"].append(accuracy)
        # Scoring the test samples draws no random numbers, so it does not alter training.
        logits.append(test_logits(model, test_batches, device))
        if calibration is not None and epoch + 1 >= config.calibration_warmup_epochs:
            calibration.refresh(logits[-1])
        if calibration is not None and labels is not None:
            state = calibration.summary(labels)
            history["calibration_labelled_fraction"].append(state["labelled_fraction"])
            history["calibration_target_accuracy"].append(state["target_accuracy"])
    synchronize(device)
    history["training_seconds"] = time.perf_counter() - start
    if calibration is None:
        del history["calibration_labelled_fraction"], history["calibration_target_accuracy"]
    return model, torch.stack(logits), history


def save_model(model, path: Path, seed: int, model_config: ModelConfig, data_config: DataConfig,
               config: TrainConfig) -> None:
    """The model after its final epoch, with the settings needed to rebuild it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict={name: value.cpu() for name, value in model.state_dict().items()}, seed=seed,
                    model_config=asdict(model_config), data_config=asdict(data_config),
                    train_config=asdict(config)), path)


def run_fold(training, testing, name: str, data_config: DataConfig, model_config: ModelConfig, config: TrainConfig,
             device, save_to: Path | None = None) -> dict:
    """Train the ensemble on `training` graphs and score the `testing` graphs.

    The result is the score after the final epoch. For comparison with work
    that reports this way, `highest_epoch` holds the metrics of the epoch with
    the highest test accuracy (chosen with the test labels, so biased upward;
    nothing is selected or reloaded).

    Also recorded: the test loss and accuracy of the ensemble after every
    epoch, the history of every model (train_model), the training time, the
    peak GPU memory, and, under "embeddings" (not written to the JSON), the
    features of the test samples computed by the first model after its final
    epoch.

    `save_to`: a path without extension; every model of the ensemble is saved as `<save_to>_model<k>.pt`.
    """
    train_loader = DataLoader(training, batch_size=config.batch_size, shuffle=True)
    # The test batches are made once, before any model is seeded: creating a
    # loader iterator draws a random number, which would alter training.
    test_batches = list(DataLoader(testing, batch_size=config.batch_size, shuffle=False))
    labels = torch.cat([data.y for data in test_batches]).tolist()
    # Every model of the ensemble calibrates on its own predictions.
    calibrations = [TargetCalibration(training, testing, config) if config.target_calibration else None
                    for _ in range(config.ensemble)]
    member_logits, histories = [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for member in range(config.ensemble):
        model, logits, history = train_model(train_loader, test_batches, calibrations[member], config.seed + member,
                                             model_config, config, device,
                                             f"{name}, model {member + 1}/{config.ensemble}", labels)
        member_logits.append(logits)
        history["test_loss"] = [float(F.cross_entropy(epoch_logits, torch.tensor(labels))) for epoch_logits in logits]
        history["test_accuracy"] = [float((epoch_logits.argmax(1) == torch.tensor(labels)).float().mean())
                                    for epoch_logits in logits]
        histories.append(history)
        if member == 0:
            features = embeddings(model, test_batches, device)
        if save_to is not None:
            save_model(model, save_to.with_name(f"{save_to.name}_model{member + 1}.pt"), config.seed + member,
                       model_config, data_config, config)
    logits = torch.stack(member_logits)                       # [models, epochs, samples, classes]
    per_epoch, test_loss = [], []
    for epoch in range(logits.shape[1]):
        predictions, scores = ensemble_scores(logits[:, epoch], config.center_logits)
        per_epoch.append((predictions.tolist(), scores.tolist()))
        # Loss of the ensemble's decision: averaged probabilities, before centring.
        probability = logits[:, epoch].softmax(dim=-1).mean(dim=0)
        test_loss.append(float(F.nll_loss(probability.clamp(min=1e-12).log(), torch.tensor(labels))))
    accuracies = [classification_metrics(labels, predictions)["accuracy"] for predictions, _ in per_epoch]
    best = accuracies.index(max(accuracies))
    predictions, scores = per_epoch[-1]
    result = dict(classification_metrics(labels, predictions, scores), training_samples=len(training),
                  testing_samples=len(testing),
                  highest_epoch=dict(classification_metrics(labels, *per_epoch[best]), epoch=best + 1),
                  test_accuracy_per_epoch=accuracies, test_loss_per_epoch=test_loss,
                  labels=labels, predictions=predictions, scores=scores,
                  highest_epoch_predictions=per_epoch[best][0], highest_epoch_scores=per_epoch[best][1],
                  training_seconds=sum(history["training_seconds"] for history in histories),
                  peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  history=histories, embeddings=features)
    if config.target_calibration:
        result["calibration"] = [calibration.summary(labels) for calibration in calibrations]
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
                note="Training time of all models of a fold, including scoring the test samples after every epoch; "
                     "graph building is not included.")


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
        training = [graph for subject in data_config.subjects if subject != held_out for graph in graphs[subject]]
        results[str(held_out)] = run_fold(training, graphs[held_out], f"Held-out S{held_out}", data_config,
                                          model_config, config, device,
                                          models_dir / f"S{held_out}" if models_dir else None)
        fold = results[str(held_out)]
        print(f"S{held_out}: {format_metrics(fold)} (highest test epoch: acc {fold['highest_epoch']['accuracy']:.4f})",
              flush=True)
    summary = summarize(results)
    summary["highest_epoch"] = summarize({k: v["highest_epoch"] for k, v in results.items()})
    print_summary(f"Mean over {len(results)} held-out subjects", summary)
    features = {subject: fold.pop("embeddings") for subject, fold in results.items()}
    return dict(protocol="leave_one_subject_out",
                evaluation="The result is the score after the final epoch. highest_epoch is the score of the epoch "
                           "with the highest test accuracy, chosen with the test labels; it is biased upward and "
                           "recorded only for comparison with work that reports this way.",
                summary=summary, subjects=results,
                unlabeled_test_subject_use=dict(
                    alignment="Euclidean alignment uses the held-out subject's own samples",
                    center_logits=config.center_logits, target_calibration=config.target_calibration,
                    note="All use the held-out subject's samples without their labels."),
                cost=run_cost(list(results.values())), embeddings=features,
                **run_information(data_config, model_config, config, device, graphs[held_out_subjects[0]]))
