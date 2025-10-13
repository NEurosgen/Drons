from __future__ import annotations

from typing import Mapping

import torch.nn as nn
from torchvision import models

from src.models.base_module import BaseLitModule


EFFICIENTNET_VARIANTS: Mapping[str, tuple] = {
    "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights),
    "b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights),
    "b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights),
    "b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights),
    "v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights),
    "v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights),
}


def _build_efficientnet(model_cfg: Mapping[str, object] | None, num_classes: int) -> nn.Module:
    cfg = dict(model_cfg or {})
    variant = str(cfg.get("variant", "b0")).lower()
    if variant not in EFFICIENTNET_VARIANTS:
        raise ValueError(f"Unsupported EfficientNet variant: {variant}")
    constructor, weight_enum = EFFICIENTNET_VARIANTS[variant]
    pretrained = bool(cfg.get("pretrained", True))
    weights = weight_enum.DEFAULT if pretrained else None
    model = constructor(weights=weights)

    classifier = model.classifier
    last_idx = -1
    if isinstance(classifier[last_idx], nn.Linear):
        in_features = classifier[last_idx].in_features
        classifier[last_idx] = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("Expected final classifier layer to be nn.Linear")

    dropout = cfg.get("dropout")
    if dropout is not None and hasattr(classifier[0], "p"):
        classifier[0] = nn.Dropout(p=float(dropout))

    return model


class LitEfficientNet(BaseLitModule):
    def __init__(self, cfg, model_cfg, num_class: int, class_weights=None):
        super().__init__(cfg, model_cfg, num_class, _build_efficientnet, class_weights=class_weights)

