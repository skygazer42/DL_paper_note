from __future__ import annotations

from .base import NumpyModelBase


class MnasNet(NumpyModelBase):
    model_id = "mnasnet"


MODEL_ID = "mnasnet"
MODEL_CLASS = MnasNet
