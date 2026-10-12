"""Model sizes. They are those of version 23; only the input size follows the dataset."""
from dataclasses import dataclass


PATHS = ("graph", "wide", "temporal")
NODE_SIGNAL_MODES = ("off", "power", "timed", "mixed")
READOUT_MODES = ("mean", "electrode")


@dataclass(frozen=True)
class ModelConfig:
    # ---- input / output -----------------------------------------------------
    num_electrodes: int = 29    # every graph lists the same electrodes in the same order
    node_features: int = 6      # band powers per electrode
    edge_features: int = 2      # phase-locking value and covariance per electrode pair
    signal_samples: int = 1000  # analysed samples per electrode
    num_classes: int = 2
    low_samples: int = 0        # samples of the low-frequency signal per electrode; 0 = no low-frequency path
    # The paths whose logits are added. All three is the model; fewer is an ablation.
    paths: tuple[str, ...] = ("graph", "wide", "temporal")

    # ---- graph head ---------------------------------------------------------
    hidden: int = 32            # width of the node stream and of the pair stream
    layers: int = 3             # hybrid blocks; version 23 has 2 (--layers 2)
    heads: int = 4
    dropout: float = 0.3
    attention_dropout: float = 0.2
    attention_clamp: float = 5.0   # attention logits are clamped to [-clamp, clamp]
    random_walk_steps: int = 8     # length of the random-walk encoding

    # Node features learned from the electrode's signal, added to the band powers in the graph head:
    #   "off"    none: the graph head of version 23
    #   "power"  the log power of every learned temporal filter over the whole window
    #   "timed"  the same, resolved in time (one value per filter and time step)
    #   "mixed"  as "timed", but before the power is taken every node mixes the filtered signals of
    #            all nodes with weights that depend on the pair of electrodes and on the trial's edge
    #            features (signal_mix_heads mixtures per node): spatial filtering done on the graph
    # The filters use temporal_filters and temporal_kernel, the time steps power_pool and power_stride.
    node_signal: str = "off"
    signal_mix_heads: int = 2
    # Transformer blocks over the time steps of every node's signal features (the temporal
    # transformer of EEGformer); 0 = none. Needs node_signal "timed" or "mixed".
    node_time_attention: int = 0
    node_time_heads: int = 4
    # How the graph head reads the nodes out:
    #   "mean"       the mean over the electrodes (and over the pairs): version 23
    #   "electrode"  in addition, readout_dim values of every electrode, kept in the electrode's place
    readout: str = "mean"
    readout_dim: int = 4

    # ---- wide path ----------------------------------------------------------
    wide_dropout: float = 0.5

    # ---- temporal encoder ---------------------------------------------------
    temporal_filters: int = 8
    temporal_kernel: int = 25      # samples (0.1 s at 250 Hz)
    spatial_filters: int = 16
    power_pool: int = 75           # samples averaged per power estimate
    power_stride: int = 15
    temporal_dropout: float = 0.7

    # ---- low-frequency path (only with low_samples > 0) -----------------------
    low_filters: int = 8           # spatial filters on the low-frequency signal
    low_dropout: float = 0.5

    def __post_init__(self):
        if self.node_time_attention < 0 or (self.node_time_attention and self.node_signal not in ("timed", "mixed")):
            raise ValueError('node_time_attention needs node_signal "timed" or "mixed"')
        if self.layers < 1:
            raise ValueError("layers must be at least 1")
        if self.readout not in READOUT_MODES:
            raise ValueError(f"readout must be one of {READOUT_MODES}")
        if self.node_signal not in NODE_SIGNAL_MODES:
            raise ValueError(f"node_signal must be one of {NODE_SIGNAL_MODES}")
        if not self.paths or len(set(self.paths)) != len(self.paths) or not set(self.paths) <= set(PATHS):
            raise ValueError(f"paths must be a non-empty selection of {PATHS} without repeats")

    @classmethod
    def for_data(cls, data_config, **overrides) -> "ModelConfig":
        """The model for the electrodes and the window length of a data configuration."""
        extra = 1 if data_config.tangent_features else 0
        return cls(**{**dict(num_electrodes=data_config.num_channels, signal_samples=data_config.window_samples,
                             node_features=6 + extra, edge_features=2 + extra,
                             low_samples=data_config.low_samples), **overrides})
