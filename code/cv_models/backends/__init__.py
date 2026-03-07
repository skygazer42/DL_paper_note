from __future__ import annotations

from typing import Literal

BackendName = Literal["numpy", "torch", "tf"]


def get_backend(name: BackendName):
    if name == "numpy":
        from .numpy_ops import NumpyOps

        return NumpyOps()
    if name == "torch":
        from .torch_ops import TorchOps

        return TorchOps()
    if name == "tf":
        from .tf_ops import TensorFlowOps

        return TensorFlowOps()
    raise ValueError(f"Unknown backend: {name!r}")

