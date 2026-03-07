from __future__ import annotations

from .base import TensorFlowModelBase


class ResNeXt(TensorFlowModelBase):
    model_id = "resnext"


MODEL_ID = "resnext"
MODEL_CLASS = ResNeXt
