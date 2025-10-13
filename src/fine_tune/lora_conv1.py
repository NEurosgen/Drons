from __future__ import annotations

from typing import Iterable, Optional

import fnmatch

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """LoRA wrapper for 1x1 Conv2d layers with groups=1."""

    def __init__(self, base_conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_conv, nn.Conv2d):
            raise TypeError("LoRAConv2d expects an nn.Conv2d base layer")
        if base_conv.kernel_size != (1, 1) or base_conv.groups != 1:
            raise ValueError("LoRAConv2d supports only 1x1 convolutions with groups=1")

        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_c = base_conv.in_channels
        out_c = base_conv.out_channels

        self.lora_A = nn.Conv2d(in_c, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, out_c, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # noqa: D401 - standard module forward
        base = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + delta

    @torch.no_grad()
    def merge_to_base_(self):
        merged = torch.matmul(
            self.lora_B.weight.view(self.lora_B.out_channels, self.lora_B.in_channels),
            self.lora_A.weight.view(self.lora_A.in_channels, self.lora_A.out_channels),
        ).t().contiguous()
        merged = merged.view(self.base.out_channels, self.base.in_channels, 1, 1)
        self.base.weight += merged * self.scaling
        self.lora_A.weight.zero_()
        self.lora_B.weight.zero_()


class LoRALinear(nn.Module):
    """LoRA adapter for linear layers."""

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear expects an nn.Linear base layer")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_f = base_linear.in_features
        out_f = base_linear.out_features

        self.lora_A = nn.Linear(in_f, r, bias=False)
        self.lora_B = nn.Linear(r, out_f, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # noqa: D401 - standard module forward
        base = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + delta

    @torch.no_grad()
    def merge_to_base_(self):
        merged = torch.matmul(self.lora_B.weight, self.lora_A.weight)
        self.base.weight += merged * self.scaling
        self.lora_A.weight.zero_()
        self.lora_B.weight.zero_()


def wrap_layers_with_lora(
    root: nn.Module,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Optional[Iterable[str]] = None,
    exclude_modules: Optional[Iterable[str]] = None,
    enable_conv1x1: bool = True,
    enable_linear: bool = True,
) -> int:
    """Replace selected layers with their LoRA counterparts.

    Returns the number of wrapped layers.
    """

    target_patterns = list(target_modules) if target_modules else None
    exclude_patterns = list(exclude_modules) if exclude_modules else []

    def _matches(name: str) -> bool:
        if target_patterns is not None:
            ok = any(fnmatch.fnmatch(name, pat) or pat in name for pat in target_patterns)
            if not ok:
                return False
        if exclude_patterns:
            if any(fnmatch.fnmatch(name, pat) or pat in name for pat in exclude_patterns):
                return False
        return True

    wrapped = 0

    for module_name, module in list(root.named_modules()):
        parent = root
        if module is root:
            continue
        parts = module_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]

        if enable_conv1x1 and isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.groups == 1:
            if _matches(module_name):
                setattr(parent, attr_name, LoRAConv2d(module, r=r, alpha=alpha, dropout=dropout))
                wrapped += 1
                continue

        if enable_linear and isinstance(module, nn.Linear):
            if _matches(module_name):
                setattr(parent, attr_name, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
                wrapped += 1

    return wrapped

