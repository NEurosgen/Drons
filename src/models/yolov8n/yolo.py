from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from ultralytics import YOLO

from ..base_module import BaseLitModule


def _replace_last_linear(model: nn.Module, out_features: int) -> None:
    last_linear = None
    last_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = module
            last_name = name
    if last_linear is None or last_name is None:
        raise RuntimeError("YOLO classifier does not expose a final nn.Linear layer")

    parent = model
    parts = last_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], nn.Linear(last_linear.in_features, out_features))


def _build_yolo(model_cfg: Mapping[str, object] | None, num_classes: int) -> nn.Module:
    cfg = dict(model_cfg or {})
    weights = cfg.get("weights")
    pretrained = bool(cfg.get("pretrained", True))
    config_path = cfg.get("config", "yolov8s-cls.yaml")

    if weights:
        yolo = YOLO(weights)
    elif pretrained:
        yolo = YOLO(cfg.get("default_weights", "yolov8s-cls.pt"))
    else:
        yolo = YOLO(config_path)

    model = yolo.model
    _replace_last_linear(model, num_classes)
    return model


class LitYOLOCls(BaseLitModule):
    def __init__(self, cfg, model_cfg, num_class: int, class_weights=None):
        super().__init__(cfg, model_cfg, num_class, _build_yolo, class_weights=class_weights)

    def forward(self, batch):
        logits = self.model(batch)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if hasattr(logits, "logits"):
            logits = logits.logits
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected YOLO to return Tensor logits, got {type(logits)}")
        return logits

