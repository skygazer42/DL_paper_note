from __future__ import annotations

from .base import PyTorchModelBase


class ResNeXt(PyTorchModelBase):
    model_id = "resnext"


MODEL_ID = "resnext"
MODEL_CLASS = ResNeXt
