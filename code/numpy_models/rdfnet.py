from __future__ import annotations

from .base import NumpyModelBase


class RDFNet(NumpyModelBase):
    model_id = "rdfnet"


MODEL_ID = "rdfnet"
MODEL_CLASS = RDFNet
