"""The classifier: graph head + wide path + temporal encoder."""
from .config import ModelConfig
from .layers import GatedGCN, GraphLayer, PairAttention, TemporalEncoder
from .model import EEGGraphModel

__all__ = ["ModelConfig", "EEGGraphModel", "GraphLayer", "GatedGCN", "PairAttention", "TemporalEncoder"]
