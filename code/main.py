
import argparse
import importlib
import sys
from pathlib import Path


def _ensure_code_on_path() -> None:
    # 允许直接 `python code/main.py ...` 运行，而不要求用户先手动设置 PYTHONPATH。
    code_dir = Path(__file__).resolve().parent
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))


def main(argv: list[str] | None = None) -> int:
    _ensure_code_on_path()
    from cv_models.sample_inputs import make_sample_inputs

    parser = argparse.ArgumentParser(description='Run one model forward (NumPy / PyTorch / TensorFlow).')
    parser.add_argument('--backend', required=True, help='numpy | pytorch | tensorflow (aliases: torch, tf)')
    parser.add_argument('--model', required=True, help='model_id (e.g. resnet, faster_r_cnn, unet)')
    args = parser.parse_args(argv)

    backend = args.backend.strip().lower()
    model_id = args.model.strip()

    # 统一把命令行里的后端别名映射到具体的代码目录。
    backend_pkg = {
        'numpy': 'numpy_models',
        'pytorch': 'pytorch_models',
        'torch': 'pytorch_models',
        'tensorflow': 'tensorflow_models',
        'tf': 'tensorflow_models',
    }.get(backend)
    if backend_pkg is None:
        raise SystemExit(f'Unknown backend: {backend!r}')

    # 每个模型文件都暴露同样的 `MODEL_CLASS` 入口，方便这里做动态加载。
    mod = importlib.import_module(f'{backend_pkg}.{model_id}')
    model = mod.MODEL_CLASS()
    outputs = model.forward(make_sample_inputs(model_id))

    print(f'backend={backend} model={model_id} outputs:')
    for k, v in outputs.items():
        shape = getattr(v, 'shape', None)
        print(f' - {k}: shape={shape}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
