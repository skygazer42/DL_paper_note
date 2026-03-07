from __future__ import annotations

from .base import PyTorchModelBase


class Metapath2vec(PyTorchModelBase):
    model_id = "metapath2vec"


MODEL_ID = "metapath2vec"
MODEL_CLASS = Metapath2vec
