from __future__ import annotations

from .base import TensorFlowModelBase


class RDFNet(TensorFlowModelBase):
    model_id = "rdfnet"


MODEL_ID = "rdfnet"
MODEL_CLASS = RDFNet
