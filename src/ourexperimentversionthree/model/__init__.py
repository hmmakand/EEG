"""Graph neural-network model for the broadcast-11 Liu2024 experiment."""

from .gcn import EEGGCN1, EEGGCN1_INPUT_FEATURES, EEGGCN1_NODES, EEGGCN1Config

__all__ = ["EEGGCN1", "EEGGCN1_INPUT_FEATURES", "EEGGCN1_NODES", "EEGGCN1Config"]
