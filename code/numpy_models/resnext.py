from __future__ import annotations

from .base import NumpyModelBase


class ResNeXt(NumpyModelBase):
    model_id = "resnext"


MODEL_ID = "resnext"
MODEL_CLASS = ResNeXt
