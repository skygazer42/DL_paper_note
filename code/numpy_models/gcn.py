from __future__ import annotations

from .base import NumpyModelBase


class GCN(NumpyModelBase):
    model_id = "gcn"


MODEL_ID = "gcn"
MODEL_CLASS = GCN
