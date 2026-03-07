from __future__ import annotations

from .base import PyTorchModelBase


class Node2vec(PyTorchModelBase):
    model_id = "node2vec"


MODEL_ID = "node2vec"
MODEL_CLASS = Node2vec
