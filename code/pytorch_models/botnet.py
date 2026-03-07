from __future__ import annotations

from .base import PyTorchModelBase


class BotNet(PyTorchModelBase):
    model_id = "botnet"


MODEL_ID = "botnet"
MODEL_CLASS = BotNet
