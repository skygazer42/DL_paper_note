# Code: Forward-Only Model Zoo (NumPy / TensorFlow / PyTorch)

`code/` 提供一个“只 forward”的 toy 版实现：把根目录 `README.md` 里出现的模型都做成可运行的 forward，并提供 3 个后端：

- `numpy`（尽量只依赖 numpy）
- `torch`（PyTorch）
- `tf`（TensorFlow，需要自行安装）

不包含训练/反向传播，也不做检测/分割的完整后处理（只输出 raw heads / logits）。

## Quick Start

从仓库根目录运行：

```bash
# NumPy backend
PYTHONPATH=code python -m cv_models.tools.smoke_test --backend numpy --model all

# PyTorch backend
PYTHONPATH=code python -m cv_models.tools.smoke_test --backend torch --model all

# TensorFlow backend (需要先安装 tensorflow)
PYTHONPATH=code python -m cv_models.tools.smoke_test --backend tf --model all
```

查看模型列表：

```bash
PYTHONPATH=code python -m cv_models.tools.smoke_test --list
```

对比不同后端的 forward（需要安装对应框架）：

```bash
PYTHONPATH=code python -m cv_models.tools.compare_backends --model resnet --backends numpy,torch
```

## 目录结构

每个模型一个文件，并且每个文件是自包含实现（不依赖仓库内其它模块的 import）：

- NumPy：`code/numpy_models/<model_id>.py`
- PyTorch：`code/pytorch_models/<model_id>.py`
- TensorFlow：`code/tensorflow_models/<model_id>.py`

例如（以 `resnet` 为例）：

```python
from numpy_models.resnet import ResNet

m = ResNet()
out = m.forward({"image": ...})
```

也可以用一个统一入口跑 forward：

```bash
python code/main.py --backend numpy --model resnet
python code/main.py --backend torch --model faster_r_cnn
```

## Notes

- 输入张量约定：
  - 图像类任务：`inputs["image"]` 为 **NHWC**（`(N,H,W,C)`）float32
  - 图模型：`inputs["x"]` / `inputs["adj"]` 或 `inputs["src"]` / `inputs["dst"]`
