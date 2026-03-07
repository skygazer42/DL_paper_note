from __future__ import annotations

from .base import NumpyModelBase


class FusionNet(NumpyModelBase):
    model_id = "fusionnet"


MODEL_ID = "fusionnet"
MODEL_CLASS = FusionNet
