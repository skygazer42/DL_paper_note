
from typing import Literal

BackendName = Literal["numpy", "torch", "tf"]


def get_backend(name: BackendName):
    # 按需导入后端，避免一个命令只跑 numpy 时还强制要求安装 torch / tensorflow。
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
