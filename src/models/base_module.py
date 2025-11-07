from __future__ import annotations

from typing import Callable, Mapping, MutableMapping

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule

from ..fine_tune.strategies import setup_finetune_strategy
from ..run_utils.lighting_utils import (
    configure_optimizers,
    init,
    test_step,
    training_step,
    validation_step,
)
from torch.ao.quantization import get_default_qconfig, get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import (
    QConfigMapping,
    convert_fx,
    prepare_fx,
    prepare_qat_fx,
)


class BaseLitModule(LightningModule):
    """Lightning module with shared training/validation logic and fine-tuning hooks."""

    def __init__(
        self,
        cfg,
        model_cfg,
        num_class: int,
        backbone_builder: Callable[[dict | None, int], object],
        *,
        class_weights=None,
    ) -> None:
        super().__init__()

        def _factory(cfg_, num_class_):
            return backbone_builder(model_cfg, num_class_)

        init(self, cfg, num_class=num_class, create_model=_factory, class_weights=class_weights)
        self._finetune_controller = setup_finetune_strategy(self.model, cfg, model_cfg)

        self._quant_cfg = self._resolve_quant_config(getattr(cfg, "quantization", None))
        self._quant_enabled = bool(self._quant_cfg.get("enabled", False))
        self._quant_prepared = False
        self._convert_on_fit_end = bool(self._quant_cfg.get("convert_on_fit_end", True))

        if self._quant_enabled and self._quant_cfg.get("prepare_on_init", True):
            self.prepare_quantization()

    # Lightning API -----------------------------------------------------
    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):  # noqa: D401 - delegated implementation
        return training_step(self, batch)

    def validation_step(self, batch, batch_idx):  # noqa: D401 - delegated implementation
        return validation_step(self, batch=batch)

    def test_step(self, batch, batch_idx):  # noqa: D401 - delegated implementation
        return test_step(self, batch)

    def configure_optimizers(self):  # noqa: D401 - delegated implementation
        return configure_optimizers(self)

    # Fine-tuning controller -------------------------------------------
    def on_train_epoch_start(self) -> None:
        controller = getattr(self, "_finetune_controller", None)
        if controller is not None:
            controller.step(self.current_epoch)

    # Quantization ------------------------------------------------------
    @staticmethod
    def _resolve_quant_config(raw_cfg) -> MutableMapping[str, object]:
        if raw_cfg is None:
            return {}
        if isinstance(raw_cfg, bool):
            return {"enabled": raw_cfg}
        if isinstance(raw_cfg, DictConfig):
            raw_cfg = OmegaConf.to_container(raw_cfg, resolve=True)
        if isinstance(raw_cfg, Mapping):
            return dict(raw_cfg)
        raise TypeError("quantization config must be a mapping or boolean")

    def prepare_quantization(self) -> None:
        if not self._quant_enabled:
            return

        backend = str(self._quant_cfg.get("backend", "fbgemm"))
        torch.backends.quantized.engine = backend

        mode = str(self._quant_cfg.get("mode", "qat")).lower()
        if mode not in {"qat", "ptq"}:
            raise ValueError(
                f"Unsupported quantization mode '{mode}'. Use 'qat' (quantization-aware training)"
                " or 'ptq' (post-training static)."
            )

        example_shape = self._quant_cfg.get("example_input_shape") or self._quant_cfg.get(
            "input_shape"
        )
        if example_shape is None:
            example_shape = (1, 3, 224, 224)
        example_input = torch.randn(*example_shape)

        if mode == "qat":
            qconfig = get_default_qat_qconfig(backend)
            prepare_fn = prepare_qat_fx
        else:
            qconfig = get_default_qconfig(backend)
            prepare_fn = prepare_fx

        qmap = QConfigMapping().set_global(qconfig)
        self.model = prepare_fn(self.model, qconfig_mapping=qmap, example_inputs=example_input)
        self._quant_prepared = True

    def convert_to_quantized(self) -> None:
        if not self._quant_prepared:
            return
        self.model = convert_fx(self.model)
        self._quant_prepared = False

    def on_fit_end(self) -> None:
        super().on_fit_end()
        if self._quant_enabled and self._convert_on_fit_end:
            self.convert_to_quantized()
