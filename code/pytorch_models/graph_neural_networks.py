from __future__ import annotations

from .base import PyTorchModelBase


class GraphNeuralNetworks(PyTorchModelBase):
    model_id = "graph_neural_networks"


MODEL_ID = "graph_neural_networks"
MODEL_CLASS = GraphNeuralNetworks
