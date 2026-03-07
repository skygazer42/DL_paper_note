# Code: Forward-Only Model Zoo (NumPy / TensorFlow / PyTorch)

本仓库原本只有论文链接与笔记索引。`code/` 目录新增了一个 **“A 档”** 实现：**只实现结构与 forward**，用于学习/查阅与快速 smoke test。

特点：
- **每个 README 里出现的模型名**都对应一个可运行的 toy 版本
- 每个模型都支持 3 个后端：`numpy` / `tf` / `torch`
- **不做训练、不做指标复现、不做检测/分割后处理**（检测/分割只输出 raw heads / logits）

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

## 每个模型一个文件（class 结构）

为了方便“按模型查代码”，另外生成了三套按后端分目录的 class 包装：

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
- 这是为了覆盖面与可运行性做的简化实现；如果你希望逐篇论文做更严格的训练复现（B/C 档），建议单独开一个 `reproduce/` 目录按论文逐个推进。
