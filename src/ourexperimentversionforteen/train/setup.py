"""Device, random initialization, model, optimizer, and loss construction."""
import torch
from .config import TrainingConfig

if __package__ == "train":
    from model.gat import GAT
else:
    from ..model.gat import GAT


def select_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if preference == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(preference)


def seed_fold(seed: int) -> None:
    # Keep the original reset before loader and model construction.
    torch.manual_seed(seed)


def build_model(in_channels: int, device: torch.device, config: TrainingConfig):
    return GAT(hidden_channels=config.hidden_channels, heads=config.heads,
               in_channels=in_channels).to(device)


def build_optimizer(model, config: TrainingConfig):
    if config.optimizer != "adam":
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


def build_criterion(config: TrainingConfig):
    if config.loss != "cross_entropy":
        raise ValueError(f"Unknown loss: {config.loss}")
    return torch.nn.CrossEntropyLoss()
