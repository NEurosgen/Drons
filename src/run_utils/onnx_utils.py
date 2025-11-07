from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence

import torch

try:  # pragma: no cover - optional dependency
    from onnxruntime.quantization import (  # type: ignore
        CalibrationDataReader,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
except Exception:  # pragma: no cover - handled at runtime
    CalibrationDataReader = object  # type: ignore
    QuantType = object  # type: ignore
    quantize_dynamic = None  # type: ignore
    quantize_static = None  # type: ignore


@dataclass
class ExportResult:
    """Container describing the result of an ONNX export."""

    onnx_path: Path
    quantized_path: Optional[Path] = None


class TorchCalibrationDataReader(CalibrationDataReader):  # type: ignore[misc]
    """Wrap a PyTorch dataloader so it can be used for static ONNX quantization."""

    def __init__(self, dataloader: Iterable, input_name: str = "input") -> None:
        if CalibrationDataReader is object:  # pragma: no cover - runtime guard
            raise RuntimeError(
                "onnxruntime is not available - install onnxruntime to use quantization"
            )
        self._dataloader = dataloader
        self._input_name = input_name
        self._iterator: Optional[Iterator] = None

    # onnxruntime quantization API expects these two methods -----------------
    def get_next(self) -> Optional[MutableMapping[str, torch.Tensor]]:  # type: ignore[override]
        if self._iterator is None:
            self._iterator = iter(self._dataloader)
        try:
            batch = next(self._iterator)
        except StopIteration:
            return None

        if isinstance(batch, Mapping):
            inputs = batch.get("inputs") or batch.get("input") or next(iter(batch.values()))
        elif isinstance(batch, Sequence):
            inputs = batch[0]
        else:
            inputs = batch

        if not isinstance(inputs, torch.Tensor):
            raise TypeError(
                "Calibration dataloader must yield torch.Tensor batches or (tensor, label) tuples"
            )

        return {self._input_name: inputs.detach().cpu().numpy()}

    def rewind(self) -> None:  # type: ignore[override]
        self._iterator = iter(self._dataloader)


def _prepare_dynamic_axes(
    input_names: Sequence[str],
    output_names: Sequence[str],
    dynamic_batch: bool,
) -> Optional[dict]:
    if not dynamic_batch:
        return None

    axes = {name: {0: "batch"} for name in input_names}
    axes.update({name: {0: "batch"} for name in output_names})
    return axes


def export_model_to_onnx(
    model: torch.nn.Module,
    output_path: Path | str,
    *,
    input_shape: Sequence[int] = (1, 3, 224, 224),
    opset_version: int = 17,
    dynamic_batch: bool = True,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    device: str | torch.device = "cpu",
) -> Path:
    """Export ``model`` to ONNX format.

    Parameters
    ----------
    model:
        Torch module to export. It must be in evaluation mode and reside on CPU/GPU.
    output_path:
        Path where the ONNX graph will be stored.
    input_shape:
        Shape of the dummy input used during tracing.
    opset_version:
        ONNX opset version.
    dynamic_batch:
        If ``True``, mark the batch dimension as dynamic in the exported graph.
    input_names / output_names:
        Custom input/output names for ONNX graph. Defaults to ``["input"]`` and
        ``["logits"]`` respectively.
    device:
        Device on which the export will be executed.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    example = torch.randn(*input_shape, device=device)

    _input_names = list(input_names) if input_names is not None else ["input"]
    _output_names = list(output_names) if output_names is not None else ["logits"]

    dynamic_axes = _prepare_dynamic_axes(_input_names, _output_names, dynamic_batch)

    torch.onnx.export(
        model,
        example,
        str(path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=_input_names,
        output_names=_output_names,
        dynamic_axes=dynamic_axes,
    )

    return path


def _normalize_weight_type(weight_type: str) -> "QuantType":
    if QuantType is object:  # pragma: no cover - runtime guard
        raise RuntimeError("onnxruntime is not available - install it to quantize models")

    weight_type_upper = weight_type.strip().upper()
    if weight_type_upper in ("QINT8", "INT8"):
        return QuantType.QInt8
    if weight_type_upper in ("QUINT8", "UINT8"):
        return QuantType.QUInt8
    raise ValueError(
        f"Unsupported weight type '{weight_type}'. Use 'qint8' or 'quint8'."
    )


def quantize_onnx_model(
    onnx_path: Path | str,
    *,
    output_path: Path | str | None = None,
    mode: str = "dynamic",
    weight_type: str = "qint8",
    per_channel: bool = False,
    calibration_reader: CalibrationDataReader | None = None,
) -> Path:
    """Quantize an exported ONNX model using ONNX Runtime.

    Parameters
    ----------
    onnx_path:
        Path to the ONNX model that should be quantized.
    output_path:
        Where the quantized model will be saved. If omitted, ``.quant.onnx`` suffix
        will be appended to ``onnx_path``.
    mode:
        Either ``"dynamic"`` or ``"static"``. Static mode requires a calibration reader.
    weight_type:
        Weight quantization type (``qint8`` or ``quint8``).
    per_channel:
        Whether to enable per-channel quantization (dynamic mode only).
    calibration_reader:
        Instance of :class:`CalibrationDataReader` used for static quantization.
    """

    if quantize_dynamic is None or QuantType is object:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "onnxruntime is not available - install onnxruntime to use quantization"
        )

    mode_normalized = mode.strip().lower()
    onnx_path = Path(onnx_path)
    out_path = Path(output_path) if output_path is not None else onnx_path.with_suffix(
        ".quant.onnx"
    )

    quant_type = _normalize_weight_type(weight_type)

    if mode_normalized == "dynamic":
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(out_path),
            weight_type=quant_type,
            per_channel=per_channel,
        )
    elif mode_normalized == "static":
        if calibration_reader is None:
            raise ValueError("Static quantization requires a calibration data reader")
        quantize_static(
            model_input=str(onnx_path),
            model_output=str(out_path),
            calibration_data_reader=calibration_reader,
            weight_type=quant_type,
        )
    else:
        raise ValueError("Quantization mode must be 'dynamic' or 'static'")

    return out_path


__all__ = [
    "ExportResult",
    "TorchCalibrationDataReader",
    "export_model_to_onnx",
    "quantize_onnx_model",
]
