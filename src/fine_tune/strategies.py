from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional

from omegaconf import DictConfig, OmegaConf

from .lora_conv1 import wrap_layers_with_lora


DEFAULT_TRAINABLE_PATTERNS = ["classifier*", "fc*", "head*"]


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


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_finetune_config(cfg: Any, model_cfg: Any | None = None) -> Mapping[str, Any]:
    ft_raw = _cfg_get(cfg, "finetune", None)
    if isinstance(ft_raw, str):
        data: MutableMapping[str, Any] = {"strategy": ft_raw}
    else:
        data = _to_dict(ft_raw)
        if "strategy" not in data:
            if "lora" in data:
                data["strategy"] = "lora"
            elif "progressive" in data:
                data["strategy"] = "progressive"
            elif "freeze" in data:
                data["strategy"] = "freeze"

    if "strategy" not in data:
        fallback = ft_raw if isinstance(ft_raw, str) else _cfg_get(cfg, "finetune", "none")
        data["strategy"] = fallback or "none"

    strategy = data.get("strategy", "none")

    if strategy == "lora":
        merged = {}
        merged.update(_to_dict(_cfg_get(cfg, "lora", None)))
        merged.update(_to_dict(data.get("lora", None)))
        data["lora"] = merged
    elif strategy == "freeze":
        merged = {}
        merged.update(_to_dict(_cfg_get(cfg, "freeze", None)))
        merged.update(_to_dict(data.get("freeze", None)))
        if merged:
            data["freeze"] = merged
    elif strategy == "progressive":
        merged = {}
        merged.update(_to_dict(_cfg_get(cfg, "progressive", None)))
        merged.update(_to_dict(data.get("progressive", None)))
        if merged:
            data["progressive"] = merged

    return data


def _match(name: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(name, pattern) or pattern in name:
            return True
    return False


def _freeze_all(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def _set_trainable(model, patterns: Iterable[str]) -> None:
    patterns = list(patterns)
    if not patterns:
        return
    for name, param in model.named_parameters():
        if _match(name, patterns):
            param.requires_grad = True


@dataclass
class ProgressiveStage:
    epoch: int
    patterns: List[str]


class ProgressiveUnfreezeController:
    def __init__(self, model, stages: Iterable[ProgressiveStage]):
        self.model = model
        self.stages = sorted(stages, key=lambda s: s.epoch)
        self._applied_epochs: set[int] = set()

    def step(self, epoch: int) -> None:
        for stage in self.stages:
            if stage.epoch <= epoch and stage.epoch not in self._applied_epochs:
                _set_trainable(self.model, stage.patterns)
                self._applied_epochs.add(stage.epoch)


def setup_finetune_strategy(model, cfg: Any, model_cfg: Any | None = None):
    ft_cfg = _normalize_finetune_config(cfg, model_cfg)
    strategy = str(ft_cfg.get("strategy", "none")).lower()

    if strategy in {"none", "off", "disable", "disabled", "false"}:
        return None

    if strategy == "freeze":
        freeze_cfg = _to_dict(ft_cfg.get("freeze", None))
        trainable = freeze_cfg.get("trainable_modules", freeze_cfg.get("modules", DEFAULT_TRAINABLE_PATTERNS))
        _freeze_all(model)
        _set_trainable(model, trainable or DEFAULT_TRAINABLE_PATTERNS)
        return None

    if strategy == "progressive":
        prog_cfg = _to_dict(ft_cfg.get("progressive", None))
        schedule_raw = prog_cfg.get("schedule", [])
        if not schedule_raw:
            raise ValueError("Progressive fine-tuning requires a non-empty schedule")

        stages: List[ProgressiveStage] = []
        for item in schedule_raw:
            item_dict = _to_dict(item)
            epoch = int(item_dict.get("epoch", 0))
            patterns = item_dict.get("modules") or item_dict.get("patterns")
            if not patterns:
                raise ValueError("Each progressive stage must define modules/patterns")
            if isinstance(patterns, str):
                patterns = [patterns]
            stages.append(ProgressiveStage(epoch=epoch, patterns=list(patterns)))

        initial = prog_cfg.get("initial_modules")
        if isinstance(initial, str):
            initial = [initial]
        if not initial and stages:
            min_epoch = min(stage.epoch for stage in stages)
            initial = []
            for stage in stages:
                if stage.epoch == min_epoch:
                    initial.extend(stage.patterns)

        _freeze_all(model)
        if initial:
            _set_trainable(model, initial)
        return ProgressiveUnfreezeController(model, stages)

    if strategy == "lora":
        lora_cfg = _to_dict(ft_cfg.get("lora", None))
        r = int(lora_cfg.get("r", 8))
        alpha = int(lora_cfg.get("alpha", 16))
        dropout = float(lora_cfg.get("dropout", 0.0))
        target_modules = lora_cfg.get("target_modules") or lora_cfg.get("modules")
        exclude_modules = lora_cfg.get("exclude_modules") or []
        enable_conv1x1 = bool(lora_cfg.get("conv1x1", True))
        enable_linear = bool(lora_cfg.get("linear", True))
        _freeze_all(model)
        wrap_layers_with_lora(
            model,
            r=r,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            enable_conv1x1=enable_conv1x1,
            enable_linear=enable_linear,
        )
        trainable = lora_cfg.get("trainable_modules", DEFAULT_TRAINABLE_PATTERNS)
        _set_trainable(model, trainable)
        return None

    raise ValueError(f"Unknown fine-tuning strategy: {strategy}")

