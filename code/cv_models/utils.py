
import re
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Callable


def repo_root() -> Path:
    # code/cv_models/utils.py -> code/cv_models -> code -> repo root
    return Path(__file__).resolve().parents[2]


def read_readme_model_names(readme_path: Path | None = None) -> list[str]:
    if readme_path is None:
        readme_path = repo_root() / "README.md"
    text = readme_path.read_text(encoding="utf-8")

    # 根 README 是模型清单的单一来源，registry 会从这里反向生成 model_id。
    names: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*-\s*\*\*(.+?)\*\*", line)
        if not m:
            continue
        raw = m.group(1).strip()
        raw = raw.rstrip(":").strip()
        if ":" in raw:
            raw = raw.split(":", 1)[0].strip()
        names.append(raw)
    return names


_SPECIAL_MODEL_IDS: dict[str, str] = {
    # README 里的展示名和文件名不完全一致，这里集中做一次纠偏。
    "U-Net": "unet",
    "Mask rcnn": "mask_rcnn",
    "DeepLabv3 plus": "deeplabv3_plus",
    "BiSeNet V2": "bisenet_v2",
    "ShuffleNet V2": "shufflenet_v2",
    "swin-Transfromer": "swin_transformer",
}


def model_id_from_readme_name(readme_name: str) -> str:
    if readme_name in _SPECIAL_MODEL_IDS:
        return _SPECIAL_MODEL_IDS[readme_name]

    # 其余名称统一规整成小写下划线形式，和文件名保持一致。
    s = readme_name.strip().lower()
    s = s.replace("+", " plus ")
    s = s.replace("-", " ")
    s = s.replace("/", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s


def tree_map(fn: Callable[[Any], Any], obj: Any) -> Any:
    # 对嵌套 dict/list/tuple 递归应用同一个变换，便于做跨后端张量转换。
    if isinstance(obj, dict):
        return {k: tree_map(fn, v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        mapped = [tree_map(fn, v) for v in obj]
        return type(obj)(mapped)  # preserves list/tuple
    if is_dataclass(obj):
        return fn(obj)
    return fn(obj)
