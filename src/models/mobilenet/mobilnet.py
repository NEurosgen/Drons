from __future__ import annotations

from typing import Mapping

import torch.nn as nn
from torchvision import models

from ..base_module import BaseLitModule


MOBILENET_VARIANTS: Mapping[str, tuple] = {
    "v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights),
    "v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights),
}


def _build_mobilenet(model_cfg: Mapping[str, object] | None, num_classes: int) -> nn.Module:
    cfg = dict(model_cfg or {})
    variant = str(cfg.get("variant", "v3_small")).lower()
    if variant not in MOBILENET_VARIANTS:
        raise ValueError(f"Unsupported MobileNet variant: {variant}")
    constructor, weight_enum = MOBILENET_VARIANTS[variant]
    pretrained = bool(cfg.get("pretrained", True))
    weights = weight_enum.DEFAULT if pretrained else None
    model = constructor(weights=weights)

    if variant.startswith("v3"):
        head_idx = -1
        in_features = model.classifier[head_idx].in_features
        model.classifier[head_idx] = nn.Linear(in_features, num_classes)
        dropout = cfg.get("dropout")
        if dropout is not None and len(model.classifier) >= 3:
            model.classifier[2] = nn.Dropout(p=float(dropout))
    else:
        raise ValueError(f"Unhandled MobileNet variant: {variant}")

    return model


class LitMobileNet(BaseLitModule):
    def __init__(self, cfg, model_cfg, num_class: int, class_weights=None):
        super().__init__(cfg, model_cfg, num_class, _build_mobilenet, class_weights=class_weights)

