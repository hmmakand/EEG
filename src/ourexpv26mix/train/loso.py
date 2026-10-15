"""Leave one subject out: train on the other subjects, test once on the held-out one."""
from dataclasses import asdict, replace
from importlib.metadata import version

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data import DataConfig, build_subject_graphs, liu2024_config
from ..data.roi import channel_names
from ..model import EEGGraphModel, ModelConfig
from .config import TrainConfig
from .calibration import TargetCalibration, train_epoch_calibrated
from .engine import score_epochs, test_logits, train_epoch
from .metrics import summarize


PACKAGES = ("torch", "torch-geometric", "numpy", "scipy", "scikit-learn", "mne")


def select_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(preference)


def train_model(loader, test_batches, seed: int, model_config: ModelConfig, config: TrainConfig, device,
                description="", calibration: TargetCalibration | None = None) -> torch.Tensor:
    """Train one model for the configured number of epochs; no validation, no early stopping.

    With a `calibration`, the epochs after the warm-up also train on the test
    subject's trials with the model's own checked predictions as targets.

    Returns its test logits after every epoch, [epochs, trials, classes].
    """
    torch.manual_seed(seed)
    model = EEGGraphModel(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    logits = []
    for epoch in tqdm(range(config.epochs), desc=description, leave=False, disable=not config.show_progress):
        if calibration is not None and epoch >= config.warmup_epochs:
            train_epoch_calibrated(model, loader, calibration, optimizer, criterion, device)
        else:
            train_epoch(model, loader, optimizer, criterion, device)
        logits.append(test_logits(model, test_batches, device))
        if calibration is not None and epoch + 1 >= config.warmup_epochs:
            calibration.refresh(logits[-1])
    return torch.stack(logits)


def shuffled_labels(graphs, seed: int):
    """Copies of the graphs with their labels permuted among them."""
    order = np.random.default_rng(seed).permutation(len(graphs))
    copies = [graph.clone() for graph in graphs]
    for copy, source in zip(copies, order):
        copy.y = graphs[source].y.clone()
    return copies


def run_fold(training, testing, name: str, model_config: ModelConfig, config: TrainConfig, device) -> dict:
    """Train the ensemble on `training` graphs and score the `testing` graphs.

    The result is the score after the final epoch. The score of the best test
    epoch is recorded next to it (see engine.score_epochs).
    """
    if config.shuffle_training_labels:
        training = shuffled_labels(training, config.seed)
    train_loader = DataLoader(training, batch_size=config.batch_size, shuffle=True)
    # The test batches are made once, before any model is seeded: creating a
    # loader iterator draws a random number, which would alter training.
    test_batches = list(DataLoader(testing, batch_size=config.batch_size, shuffle=False))
    labels = torch.cat([data.y for data in test_batches]).tolist()
    # Every model of the ensemble calibrates on its own predictions.
    calibrations = [TargetCalibration(training, testing, config) if config.target_calibration else None
                    for _ in range(config.ensemble)]
    logits = torch.stack([train_model(train_loader, test_batches, config.seed + member, model_config, config, device,
                                      f"{name}, model {member + 1}/{config.ensemble}", calibrations[member])
                          for member in range(config.ensemble)])
    result = dict(score_epochs(logits, labels, config.center_logits), training_trials=len(training),
                  testing_trials=len(testing))
    if config.target_calibration:
        result["calibration"] = [calibration.summary(labels) for calibration in calibrations]
    return result


EVALUATION = ("The result is the score after the final epoch; no test label is used to obtain it. `highest_epoch` "
              "is the score of the epoch with the highest test accuracy, chosen per fold with the test labels; it "
              "is optimistically biased and is recorded for comparison only.")


def run_loso(data_dir, data_config: DataConfig | None = None, model_config: ModelConfig | None = None,
             config: TrainConfig = TrainConfig(), held_out_subjects=None) -> dict:
    """Every subject in `held_out_subjects` (default: all) is the test subject of one fold."""
    data_config = data_config or liu2024_config()
    if config.target_calibration and channel_names(data_config.dataset) is None:
        print(f"Target calibration is switched off: {data_config.dataset} has no channel names", flush=True)
        config = replace(config, target_calibration=False)
    config.validate()
    model_config = model_config or ModelConfig.for_data(data_config)
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
        results[str(held_out)] = run_fold(training, graphs[held_out], f"Held-out S{held_out}",
                                          model_config, config, device)
        fold = results[str(held_out)]
        print(f"S{held_out}: accuracy {fold['accuracy']:.4f} (highest test epoch {fold['highest_epoch']['accuracy']:.4f})",
              flush=True)
    summary = summarize(results)
    summary["highest_epoch"] = summarize({subject: fold["highest_epoch"] for subject, fold in results.items()})
    print(f"Mean accuracy over held-out subjects: {summary['accuracy']:.4f} "
          f"(highest test epoch {summary['highest_epoch']['accuracy']:.4f})", flush=True)
    return dict(dataset=data_config.dataset, protocol="leave_one_subject_out", evaluation=EVALUATION,
                shuffled_training_labels=config.shuffle_training_labels,
                summary=summary, subjects=results,
                data_config=asdict(data_config), model_config=asdict(model_config), train_config=asdict(config),
                unlabeled_test_subject_use=dict(
                    alignment="Euclidean alignment uses the held-out subject's own trials",
                    center_logits=config.center_logits,
                    note="Both use the held-out subject's trials without their labels."),
                device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                packages={name: version(name) for name in PACKAGES})
