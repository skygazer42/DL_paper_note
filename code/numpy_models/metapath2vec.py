from __future__ import annotations

from .base import NumpyModelBase


class Metapath2vec(NumpyModelBase):
    model_id = "metapath2vec"


MODEL_ID = "metapath2vec"
MODEL_CLASS = Metapath2vec
