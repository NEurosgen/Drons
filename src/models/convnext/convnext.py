from __future__ import annotations

from typing import Mapping

import torch.nn as nn
from torchvision import models


from ..base_module import BaseLitModule

CONVNEXT_VARIANTS: Mapping[str, tuple] = {
    "tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights),
    "small": (models.convnext_small, models.ConvNeXt_Small_Weights),
    "base": (models.convnext_base, models.ConvNeXt_Base_Weights),
    "large": (models.convnext_large, models.ConvNeXt_Large_Weights),
}


def _build_convnext(model_cfg: Mapping[str, object] | None, num_classes: int) -> nn.Module:
    cfg = dict(model_cfg or {})
    variant = str(cfg.get("variant", "tiny")).lower()
    if variant not in CONVNEXT_VARIANTS:
        raise ValueError(f"Unsupported ConvNeXt variant: {variant}")
    constructor, weight_enum = CONVNEXT_VARIANTS[variant]
    pretrained = bool(cfg.get("pretrained", True))
    weights = weight_enum.DEFAULT if pretrained else None
    model = constructor(weights=weights)

    classifier = model.classifier
    linear_idx = -1
    if isinstance(classifier[linear_idx], nn.Linear):
        in_features = classifier[linear_idx].in_features
        classifier[linear_idx] = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError("Expected the final classifier layer to be nn.Linear")

    dropout = cfg.get("dropout")
    if dropout is not None and len(classifier) > 1 and isinstance(classifier[0], nn.LayerNorm):
        classifier.insert(1, nn.Dropout(p=float(dropout)))

    return model


class LitConvNeXt(BaseLitModule):
    def __init__(self, cfg, model_cfg, num_class: int, class_weights=None):
        super().__init__(cfg, model_cfg, num_class, _build_convnext, class_weights=class_weights)

