"""Device, random initialization, model, optimizer, and loss construction."""
import torch
from .config import TrainingConfig

if __package__ == "train":
    from model.gat import GAT
    from model.custom import GritGPSConfig, GritGPSModel
else:
    from ..model.gat import GAT
    from ..model.custom import GritGPSConfig, GritGPSModel


# Temporal encoder path used when graphs carry node signals; see
# model/custom/config.py for the choices.
TEMPORAL_MODE = "head"


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


def build_model(in_channels: int, device: torch.device, config: TrainingConfig,
                num_nodes: int | None = None, edge_dim: int | None = None,
                signal_samples: int | None = None):
    if config.model == "gat":
        return GAT(hidden_channels=config.hidden_channels, heads=config.heads,
                   in_channels=in_channels).to(device)
    if config.model != "custom":
        raise ValueError(f"Unknown model: {config.model}")
    # num_nodes enables the per-electrode embedding; every graph must then
    # list the same electrodes in the same order.
    # edge_dim makes the model read the graphs' edge features. The wide head
    # and the temporal encoder share the fixed-electrode assumption of the
    # electrode embedding; the temporal encoder also needs the node signals.
    temporal = TEMPORAL_MODE if num_nodes is not None and signal_samples is not None else "none"
    return GritGPSModel(GritGPSConfig(dim_in=in_channels, dim_out=2,
                                      num_electrodes=num_nodes,
                                      edge_dim=edge_dim,
                                      wide_head=num_nodes is not None,
                                      temporal=temporal,
                                      signal_samples=signal_samples or 0)).to(device)


class Ensemble(torch.nn.Module):
    """Average the class probabilities of independently trained models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    @property
    def uses_edge_attr(self) -> bool:
        return getattr(self.models[0], "uses_edge_attr", False) is True

    @property
    def uses_signal(self) -> bool:
        return getattr(self.models[0], "uses_signal", False) is True

    def forward(self, *inputs, **keywords):
        probabilities = torch.stack([model(*inputs, **keywords).softmax(dim=-1) for model in self.models])
        return probabilities.mean(dim=0).log()


def sample_dimensions(graph) -> dict:
    """build_model keyword arguments read from one graph of the dataset."""
    if graph.get("edge_attr") is not None:
        edge_dim = graph.edge_attr.size(1)
    else:
        edge_dim = 1 if graph.get("edge_weight") is not None else None
    signal = graph.get("signal")
    return dict(num_nodes=graph.num_nodes, edge_dim=edge_dim,
                signal_samples=signal.size(1) if signal is not None else None)


def build_optimizer(model, config: TrainingConfig):
    if config.optimizer != "adam":
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)


def build_criterion(config: TrainingConfig):
    if config.loss != "cross_entropy":
        raise ValueError(f"Unknown loss: {config.loss}")
    return torch.nn.CrossEntropyLoss()
