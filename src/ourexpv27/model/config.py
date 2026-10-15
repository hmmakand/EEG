"""Model sizes. These are the values of the reported result; none is an untested alternative."""
from dataclasses import dataclass

PATHS = ("graph", "wide", "temporal")


@dataclass(frozen=True)
class ModelConfig:
    # ---- input / output -----------------------------------------------------
    num_electrodes: int = 29    # every graph lists the same electrodes in the same order
    node_features: int = 6      # band powers per electrode
    edge_features: int = 2      # phase-locking value and covariance per electrode pair
    signal_samples: int = 475   # analysed samples per electrode (1.9 s at 250 Hz)
    num_classes: int = 2
    # The paths whose logits are added. All three is the method; fewer is an ablation.
    paths: tuple[str, ...] = PATHS

    # ---- graph head ---------------------------------------------------------
    hidden: int = 32            # width of the node stream and of the pair stream
    layers: int = 3             # hybrid blocks (GatedGCN + pair attention)
    heads: int = 4
    dropout: float = 0.3
    attention_dropout: float = 0.2
    attention_clamp: float = 5.0   # attention logits are clamped to [-clamp, clamp]
    random_walk_steps: int = 8     # length of the random-walk encoding

    # ---- wide path ----------------------------------------------------------
    wide_dropout: float = 0.5

    # ---- temporal encoder ---------------------------------------------------
    temporal_filters: int = 8
    temporal_kernel: int = 25      # samples (0.1 s at 250 Hz)
    spatial_filters: int = 16
    power_pool: int = 75           # samples averaged per power estimate
    power_stride: int = 15
    temporal_dropout: float = 0.7

    def __post_init__(self):
        if not self.paths or len(set(self.paths)) != len(self.paths) or not set(self.paths) <= set(PATHS):
            raise ValueError(f"paths must be a non-empty selection of {PATHS} without repeats")

    @classmethod
    def for_data(cls, data_config, **overrides) -> "ModelConfig":
        """The model for the electrodes and the sample length of a dataset."""
        return cls(**{**dict(num_electrodes=data_config.num_channels, signal_samples=data_config.window_samples),
                      **overrides})
