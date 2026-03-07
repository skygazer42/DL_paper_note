from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .utils import model_id_from_readme_name, read_readme_model_names

Task = Literal[
    "classification",
    "detection",
    "segmentation",
    "transformer",
    "gan",
    "graph_pair",
    "graph_adj",
    "mlp",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    readme_name: str
    task: Task


README_MODEL_NAMES: list[str] = read_readme_model_names()


_DETECTION = {
    "R-CNN",
    "Fast-R-CNN",
    "Faster R-CNN",
    "Cascade R-CNN",
    "YOLOv1",
    "YOLOv2",
    "YOLOv3",
    "YOLOv4",
    "YOLOv5",
    "PPYOLOE",
    "RT-DETR",
    "FPN",
    "RetinaNet",
    "FCOS",
    "Mask rcnn",
    "M2Det",
    "EfficientDet",
    "Cascade RCNN-RS",
}

_SEGMENTATION = {
    "FCN",
    "DeconvNet",
    "U-Net",
    "SegNet",
    "ENet",
    "FusionNet",
    "DeepLabv1",
    "DeepLabv2",
    "DeepLabv3",
    "DeepLabv3 plus",
    "GCN",
    "ExFuse",
    "DFN",
    "BiSeNetv1",
    "BiSeNet V2",
    "RDFNet",
    "RedNet",
    "DFANet",
}

_TRANSFORMERS = {
    "Transformer",
    "ViT",
    "T2T",
    "BotNet",
    "TnT",
    "MAE",
    "PVT",
    "swin-Transfromer",
    "Deit",
}

_GAN = {"GAN", "pix2pix", "CycleGAN"}

_GRAPH_PAIR = {"node2vec", "LINE", "metapath2vec"}
_GRAPH_ADJ = {"SDNE", "Graph neural networks", "A Survey on Graph Diffusion Models"}


def _task_for_readme_name(name: str) -> Task:
    # Cleanups for slightly messy README bullets.
    if name.startswith("EfficientDet"):
        name = "EfficientDet"
    if name in _DETECTION:
        return "detection"
    if name in _SEGMENTATION:
        return "segmentation"
    if name in _TRANSFORMERS:
        return "transformer"
    if name in _GAN:
        return "gan"
    if name in _GRAPH_PAIR:
        return "graph_pair"
    if name in _GRAPH_ADJ:
        return "graph_adj"
    if name == "BP":
        return "mlp"
    return "classification"


MODEL_SPECS: dict[str, ModelSpec] = {}
for _readme_name in README_MODEL_NAMES:
    clean = _readme_name
    if clean.startswith("EfficientDet"):
        clean = "EfficientDet"
    model_id = model_id_from_readme_name(clean)
    if model_id in MODEL_SPECS:
        raise ValueError(f"Duplicate model_id {model_id!r} from README name {clean!r}")
    MODEL_SPECS[model_id] = ModelSpec(model_id=model_id, readme_name=clean, task=_task_for_readme_name(clean))

