from __future__ import annotations

from typing import Mapping

import torch.nn as nn
from torchvision import models

from src.models.base_module import BaseLitModule


RESNET_VARIANTS: Mapping[str, tuple] = {
    "18": (models.resnet18, models.ResNet18_Weights),
    "34": (models.resnet34, models.ResNet34_Weights),
    "50": (models.resnet50, models.ResNet50_Weights),
}


def _build_resnet(model_cfg: Mapping[str, object] | None, num_classes: int) -> nn.Module:
    cfg = dict(model_cfg or {})
    variant_key = str(cfg.get("variant", "18")).replace("resnet", "").strip()
    if variant_key not in RESNET_VARIANTS:
        raise ValueError(f"Unsupported ResNet variant: {variant_key}")
    constructor, weight_enum = RESNET_VARIANTS[variant_key]
    pretrained = bool(cfg.get("pretrained", True))
    weights = weight_enum.DEFAULT if pretrained else None
    model = constructor(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


class LitResNet18(BaseLitModule):
    def __init__(self, cfg, model_cfg, num_class: int, class_weights=None):
        super().__init__(cfg, model_cfg, num_class, _build_resnet, class_weights=class_weights)

