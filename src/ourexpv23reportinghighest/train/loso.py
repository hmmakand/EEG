"""Leave one subject out: train on the other subjects, score the held-out one after every epoch.

Training is the training of version 23. The only addition is that the
held-out subject is scored after every epoch, so that two results can be
reported side by side: the final epoch, and the epoch with the highest test
accuracy (see engine.score_epochs).
"""
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data import DataConfig, build_subject_graphs
from ..model import EEGGraphModel, ModelConfig
from .config import TrainConfig
from .engine import score_epochs, test_logits, train_epoch
from .metrics import summarize_both


def select_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(preference)


def train_model(loader, test_batches, seed: int, model_config: ModelConfig, config: TrainConfig, device,
                description="") -> torch.Tensor:
    """Train one model for the configured number of epochs; no validation, no early stopping.

    Returns its test logits after every epoch, [epochs, trials, classes].
    """
    torch.manual_seed(seed)
    model = EEGGraphModel(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    logits = []
    for _ in tqdm(range(config.epochs), desc=description, leave=False, disable=not config.show_progress):
        train_epoch(model, loader, optimizer, criterion, device)
        logits.append(test_logits(model, test_batches, device))
    return torch.stack(logits)


def shuffled_labels(graphs, seed: int):
    """Copies of the graphs with their labels permuted among them."""
    order = np.random.default_rng(seed).permutation(len(graphs))
    copies = [graph.clone() for graph in graphs]
    for copy, source in zip(copies, order):
        copy.y = graphs[source].y.clone()
    return copies


def run_fold(training, testing, name: str, model_config: ModelConfig, config: TrainConfig, device) -> dict:
    """Train the ensemble on `training` graphs and score the `testing` graphs after every epoch."""
    if config.shuffle_training_labels:
        training = shuffled_labels(training, config.seed)
    train_loader = DataLoader(training, batch_size=config.batch_size, shuffle=True)
    # The test batches are made once, before any model is seeded: creating a
    # loader iterator draws a random number, which would alter training.
    test_batches = list(DataLoader(testing, batch_size=config.batch_size, shuffle=False))
    labels = torch.cat([data.y for data in test_batches]).tolist()
    logits = torch.stack([train_model(train_loader, test_batches, config.seed + member, model_config, config, device,
                                      f"{name}, model {member + 1}/{config.ensemble}")
                          for member in range(config.ensemble)])
    return dict(score_epochs(logits, labels, config.center_logits), training_trials=len(training),
                testing_trials=len(testing))


EVALUATION = dict(
    final_epoch="Score after the last epoch. No test label is used to obtain it.",
    highest_epoch="Score of the epoch with the highest test accuracy, chosen separately for every fold with "
                  "the test labels. Optimistically biased; not an estimate of accuracy on new data.")


def print_scores(name: str, scores: dict) -> None:
    print(f"{name}: final epoch {scores['final_epoch']['accuracy']:.4f}, "
          f"highest epoch {scores['highest_epoch']['accuracy']:.4f}", flush=True)


def run_loso(data_dir, data_config: DataConfig = DataConfig(), model_config: ModelConfig = ModelConfig(),
             config: TrainConfig = TrainConfig(), held_out_subjects=None) -> dict:
    """Every subject in `held_out_subjects` (default: all) is the test subject of one fold."""
    config.validate()
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
        print_scores(f"S{held_out}", results[str(held_out)])
    summary = summarize_both(results)
    print_scores("Mean over held-out subjects", summary)
    return dict(protocol="leave_one_subject_out", evaluation=EVALUATION,
                shuffled_training_labels=config.shuffle_training_labels,
                summary=summary, subjects=results,
                data_config=asdict(data_config), model_config=asdict(model_config), train_config=asdict(config),
                unlabeled_test_subject_use=dict(
                    alignment="Euclidean alignment uses the held-out subject's own trials",
                    center_logits=config.center_logits,
                    note="Both use the held-out subject's trials without their labels."),
                device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                packages={name: version(name) for name in ("torch", "torch-geometric", "numpy", "scipy", "scikit-learn")})
