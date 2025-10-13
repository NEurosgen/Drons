from __future__ import annotations

from typing import Callable

from pytorch_lightning import LightningModule

from src.fine_tune.strategies import setup_finetune_strategy
from src.run_utils.lighting_utils import (
    configure_optimizers,
    init,
    test_step,
    training_step,
    validation_step,
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

