from __future__ import annotations

from .base import TensorFlowModelBase


class GCN(TensorFlowModelBase):
    model_id = "gcn"


MODEL_ID = "gcn"
MODEL_CLASS = GCN
