from __future__ import annotations

from .base import NumpyModelBase


class DeconvNet(NumpyModelBase):
    model_id = "deconvnet"


MODEL_ID = "deconvnet"
MODEL_CLASS = DeconvNet
