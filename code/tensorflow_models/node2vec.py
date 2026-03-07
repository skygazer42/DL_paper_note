from __future__ import annotations

from .base import TensorFlowModelBase


class Node2vec(TensorFlowModelBase):
    model_id = "node2vec"


MODEL_ID = "node2vec"
MODEL_CLASS = Node2vec
