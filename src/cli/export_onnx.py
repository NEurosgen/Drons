from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.models.create_model import create_model
from src.run_utils.onnx_utils import export_model_to_onnx, quantize_onnx_model

log = logging.getLogger(__name__)


def _resolve_path(path_like: str | None) -> Path | None:
    if path_like in (None, "", "null"):
        return None
    return Path(to_absolute_path(str(path_like)))


def _count_classes(root: Path) -> int:
    if not root.exists():
        raise FileNotFoundError(f"Dataset split directory does not exist: {root}")
    subdirs = [entry for entry in root.iterdir() if entry.is_dir()]
    if not subdirs:
        raise RuntimeError(
            f"Unable to determine the number of classes from '{root}'. Ensure it contains sub-folders."
        )
    return len(subdirs)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    log.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if getattr(incompatible, "missing_keys", None):
        log.warning("Missing keys during checkpoint load: %s", incompatible.missing_keys)
    if getattr(incompatible, "unexpected_keys", None):
        log.warning("Unexpected keys during checkpoint load: %s", incompatible.unexpected_keys)


def _to_sequence(values) -> Sequence[str] | None:
    if values is None:
        return None
    if isinstance(values, (list, tuple)):
        return list(values)
    if isinstance(values, str):
        return [values]
    raise TypeError("Expected list/tuple/str for input/output names in onnx_export config")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if "onnx_export" not in cfg or not cfg.onnx_export.enabled:
        raise ValueError("Enable onnx_export.enabled in the config before running this command")

    export_cfg = cfg.onnx_export
    log.info("ONNX export configuration:\n%s", OmegaConf.to_yaml(export_cfg))

    dataset_root = Path(to_absolute_path(cfg.path))
    num_classes = _count_classes(dataset_root / "train")
    log.info("Detected %s classes for export", num_classes)

    lit_module = create_model(cfg, num_class=num_classes)
    checkpoint_path = _resolve_path(export_cfg.get("checkpoint"))
    if checkpoint_path:
        _load_checkpoint(lit_module, checkpoint_path)

    backbone = getattr(lit_module, "model", lit_module)

    input_shape = tuple(int(dim) for dim in export_cfg.get("input_shape", [1, 3, 224, 224]))
    opset_version = int(export_cfg.get("opset_version", 17))
    dynamic_batch = bool(export_cfg.get("dynamic_batch", True))
    device = export_cfg.get("device", "cpu")

    input_names = _to_sequence(export_cfg.get("input_names"))
    output_names = _to_sequence(export_cfg.get("output_names"))

    output_path = Path(to_absolute_path(export_cfg.get("output_path", "model.onnx")))

    log.info("Exporting model to ONNX: %s", output_path)
    export_model_to_onnx(
        backbone,
        output_path,
        input_shape=input_shape,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch,
        input_names=input_names,
        output_names=output_names,
        device=device,
    )
    log.info("ONNX model saved to %s", output_path)

    quant_cfg = export_cfg.get("quantize")
    if quant_cfg and quant_cfg.get("enabled", False):
        log.info("Starting quantization: %s", OmegaConf.to_yaml(quant_cfg))
        quant_mode = quant_cfg.get("mode", "dynamic")
        quant_output = _resolve_path(quant_cfg.get("output_path"))
        if quant_output is None:
            quant_output = output_path.with_suffix(".quant.onnx")

        per_channel = bool(quant_cfg.get("per_channel", False))
        weight_type = quant_cfg.get("weight_type", "qint8")

        calibration_reader = None
        if quant_mode.strip().lower() == "static":
            raise NotImplementedError(
                "Static quantization requires providing a calibration dataloader. "
                "Please extend the export script with a TorchCalibrationDataReader if needed."
            )

        quantize_onnx_model(
            output_path,
            output_path=quant_output,
            mode=quant_mode,
            weight_type=weight_type,
            per_channel=per_channel,
            calibration_reader=calibration_reader,
        )
        log.info("Quantized ONNX model saved to %s", quant_output)


if __name__ == "__main__":
    main()
