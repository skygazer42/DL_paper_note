from __future__ import annotations

from .base import TensorFlowModelBase


class DFN(TensorFlowModelBase):
    model_id = "dfn"


MODEL_ID = "dfn"
MODEL_CLASS = DFN
