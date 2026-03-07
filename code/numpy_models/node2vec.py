from __future__ import annotations

from .base import NumpyModelBase


class Node2vec(NumpyModelBase):
    model_id = "node2vec"


MODEL_ID = "node2vec"
MODEL_CLASS = Node2vec
