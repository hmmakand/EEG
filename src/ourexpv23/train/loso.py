"""Leave one subject out: train on the other subjects, test once on the held-out one."""
from dataclasses import asdict
from importlib.metadata import version

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..data import DataConfig, build_subject_graphs
from ..model import EEGGraphModel, ModelConfig
from .config import TrainConfig
from .engine import Ensemble, evaluate, train_epoch
from .metrics import classification_metrics, summarize


def select_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(preference)


def train_model(loader, seed: int, model_config: ModelConfig, config: TrainConfig, device, description=""):
    """One model trained for the configured number of epochs; no validation, no early stopping."""
    torch.manual_seed(seed)
    model = EEGGraphModel(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in tqdm(range(config.epochs), desc=description, leave=False, disable=not config.show_progress):
        train_epoch(model, loader, optimizer, criterion, device)
    return model


def run_fold(training, testing, name: str, model_config: ModelConfig, config: TrainConfig, device) -> dict:
    """Train the ensemble on `training` graphs and score the `testing` graphs once."""
    train_loader = DataLoader(training, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(testing, batch_size=config.batch_size, shuffle=False)
    models = [train_model(train_loader, config.seed + member, model_config, config, device,
                          f"{name}, model {member + 1}/{config.ensemble}")
              for member in range(config.ensemble)]
    model = models[0] if config.ensemble == 1 else Ensemble(models)
    labels, predictions = evaluate(model, test_loader, device, config.center_logits)
    return dict(classification_metrics(labels, predictions), training_trials=len(training),
                testing_trials=len(testing))


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
        print(f"S{held_out}: accuracy {results[str(held_out)]['accuracy']:.4f}", flush=True)
    summary = summarize(results)
    print(f"Mean accuracy over held-out subjects: {summary['accuracy']:.4f}", flush=True)
    return dict(protocol="leave_one_subject_out", evaluation="final epoch, once per held-out subject",
                summary=summary, subjects=results,
                data_config=asdict(data_config), model_config=asdict(model_config), train_config=asdict(config),
                unlabeled_test_subject_use=dict(
                    alignment="Euclidean alignment uses the held-out subject's own trials",
                    center_logits=config.center_logits,
                    note="Both use the held-out subject's trials without their labels."),
                device=str(device),
                device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                packages={name: version(name) for name in ("torch", "torch-geometric", "numpy", "scipy", "scikit-learn")})
