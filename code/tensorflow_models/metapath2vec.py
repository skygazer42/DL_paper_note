from __future__ import annotations

from .base import TensorFlowModelBase


class Metapath2vec(TensorFlowModelBase):
    model_id = "metapath2vec"


MODEL_ID = "metapath2vec"
MODEL_CLASS = Metapath2vec
