from __future__ import annotations

from typing import Any

from cv_models.build import build_model


class PyTorchModelBase:
    model_id: str

    def __init__(self, **kwargs: Any):
        self._model = build_model(self.model_id, backend="torch")

    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self._model(inputs)

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self.forward(inputs)
