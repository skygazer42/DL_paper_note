from __future__ import annotations

from .base import NumpyModelBase


class ZFNet(NumpyModelBase):
    model_id = "zfnet"


MODEL_ID = "zfnet"
MODEL_CLASS = ZFNet
