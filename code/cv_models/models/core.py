
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ForwardModel:
    model_id: str
    backend: Any
    forward_fn: Callable[[dict[str, Any]], dict[str, Any]]

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # 外部统一喂 NumPy / Python 标量；真正进模型前在这里转成对应后端张量。
        converted: dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, (int, float, str)):
                converted[k] = v
            else:
                converted[k] = self.backend.asarray(v)
        return self.forward_fn(converted)
