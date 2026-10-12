"""Model sizes. These are the values of the best result; none is an untested alternative."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # ---- input / output -----------------------------------------------------
    num_electrodes: int = 22    # every graph lists the same electrodes in the same order
    node_features: int = 6      # band powers per electrode
    edge_features: int = 2      # phase-locking value and covariance per electrode pair
    signal_samples: int = 500   # analysed samples per electrode
    num_classes: int = 2

    # ---- graph head ---------------------------------------------------------
    hidden: int = 32            # width of the node stream and of the pair stream
    layers: int = 2
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
