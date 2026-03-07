from __future__ import annotations

from .base import TensorFlowModelBase


class PPYOLOE(TensorFlowModelBase):
    model_id = "ppyoloe"


MODEL_ID = "ppyoloe"
MODEL_CLASS = PPYOLOE
