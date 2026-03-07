from __future__ import annotations

from .base import TensorFlowModelBase


class ZFNet(TensorFlowModelBase):
    model_id = "zfnet"


MODEL_ID = "zfnet"
MODEL_CLASS = ZFNet
