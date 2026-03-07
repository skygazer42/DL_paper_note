from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ForwardModel:
    model_id: str
    backend: Any
    forward_fn: Callable[[dict[str, Any]], dict[str, Any]]

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Convert NumPy inputs into backend tensors when needed. Keep simple scalars as-is.
        converted: dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, (int, float, str)):
                converted[k] = v
            else:
                converted[k] = self.backend.asarray(v)
        return self.forward_fn(converted)

