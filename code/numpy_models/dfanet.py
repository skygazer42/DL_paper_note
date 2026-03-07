from __future__ import annotations

from .base import NumpyModelBase


class DFANet(NumpyModelBase):
    model_id = "dfanet"


MODEL_ID = "dfanet"
MODEL_CLASS = DFANet
