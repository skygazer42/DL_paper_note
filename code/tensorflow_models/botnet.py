from __future__ import annotations

from .base import TensorFlowModelBase


class BotNet(TensorFlowModelBase):
    model_id = "botnet"


MODEL_ID = "botnet"
MODEL_CLASS = BotNet
