from __future__ import annotations

from .base import NumpyModelBase


class DFN(NumpyModelBase):
    model_id = "dfn"


MODEL_ID = "dfn"
MODEL_CLASS = DFN
