from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Type

from omegaconf import DictConfig, OmegaConf

from .convnext.convnext import LitConvNeXt
from .efficientnet.efficientnet import LitEfficientNet
from .mobilenet.mobilnet import LitMobileNet
from .resnet18.resnet import LitResNet18
from  .yolov8n.yolo import LitYOLOCls


MODEL_REGISTRY: Mapping[str, Type] = {
    "mobilenet": LitMobileNet,
    "mobilenet_v3": LitMobileNet,
    "mobilnet": LitMobileNet,
    "resnet": LitResNet18,
    "resnet18": LitResNet18,
    "efficientnet": LitEfficientNet,
    "convnext": LitConvNeXt,
    "yolo": LitYOLOCls,
}


def _to_dict(cfg_fragment: Any) -> MutableMapping[str, Any]:
    if cfg_fragment is None:
        return {}
    if isinstance(cfg_fragment, DictConfig):
        return OmegaConf.to_container(cfg_fragment, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg_fragment, Mapping):
        return dict(cfg_fragment)
    if hasattr(cfg_fragment, "__dict__"):
        return dict(vars(cfg_fragment))
    return {}


def _resolve_model_cfg(cfg: Any) -> MutableMapping[str, Any]:
    model_cfg = _to_dict(getattr(cfg, "model", None))
    if "name" not in model_cfg and hasattr(cfg, "name"):
        model_cfg["name"] = getattr(cfg, "name")

    for key in ("variant", "pretrained", "weights", "config", "default_weights"):
        if key not in model_cfg and hasattr(cfg, key):
            value = getattr(cfg, key)
            if value is not None:
                model_cfg[key] = value

    return model_cfg


def create_model(cfg, num_class: int, class_weights=None):
    model_cfg = _resolve_model_cfg(cfg)
    name = model_cfg.get("name")
    if not name:
        raise ValueError("Model name must be specified via cfg.name or cfg.model.name")

    key = str(name).lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture: {name}")

    lit_cls = MODEL_REGISTRY[key]
    model = lit_cls(cfg, model_cfg, num_class, class_weights=class_weights)

    quant_cfg = getattr(cfg, "quantization", None)
    enabled = False
    if isinstance(quant_cfg, bool):
        enabled = quant_cfg
    elif isinstance(quant_cfg, DictConfig):
        enabled = bool(quant_cfg.get("enabled", False))
    elif isinstance(quant_cfg, Mapping):
        enabled = bool(quant_cfg.get("enabled", False))

    already_prepared = bool(getattr(model, "_quant_prepared", False))
    if enabled and not already_prepared and hasattr(model, "prepare_quantization"):
        # BaseLitModule handles automatic preparation on init when requested, but
        # call explicitly to support custom implementations.
        model.prepare_quantization()

    return model

