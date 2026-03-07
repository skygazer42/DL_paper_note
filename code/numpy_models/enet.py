from __future__ import annotations

from .base import NumpyModelBase


class ENet(NumpyModelBase):
    model_id = "enet"


MODEL_ID = "enet"
MODEL_CLASS = ENet
