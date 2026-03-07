from __future__ import annotations

from .base import NumpyModelBase


class RedNet(NumpyModelBase):
    model_id = "rednet"


MODEL_ID = "rednet"
MODEL_CLASS = RedNet
