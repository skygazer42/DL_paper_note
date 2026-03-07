from __future__ import annotations

from .base import NumpyModelBase


class BotNet(NumpyModelBase):
    model_id = "botnet"


MODEL_ID = "botnet"
MODEL_CLASS = BotNet
