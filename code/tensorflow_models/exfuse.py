from __future__ import annotations

from .base import TensorFlowModelBase


class ExFuse(TensorFlowModelBase):
    model_id = "exfuse"


MODEL_ID = "exfuse"
MODEL_CLASS = ExFuse
