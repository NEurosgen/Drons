import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRAConv2d(nn.Module):
    """
    LoRA wrapper for 1x1 Conv2d (groups=1).
    Output = base_conv(x) + (alpha/r) * B(A(x))
    Only LoRA params (A,B) are trainable. Base conv is frozen.
    """
    def __init__(self, base_conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d)
        assert base_conv.kernel_size == (1, 1) and base_conv.groups == 1, \
            "LoRAConv2d here supports only pointwise 1x1 convs with groups=1."

        self.base = base_conv
        for p in self.base.parameters():
            p.requires_grad = False  # freeze base conv

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        in_c  = base_conv.in_channels
        out_c = base_conv.out_channels

        # A: in_c -> r (1x1), B: r -> out_c (1x1)
        self.lora_A = nn.Conv2d(in_c,  r,    kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r,     out_c, kernel_size=1, bias=False)

        # small init so the delta starts near zero
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base = self.base(x)
        delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + delta

    @torch.no_grad()
    def merge_to_base_(self):
        merged = torch.matmul(
            self.lora_B.weight.view(self.lora_B.out_channels, self.lora_B.in_channels),
            self.lora_A.weight.view(self.lora_A.in_channels, self.lora_A.out_channels)
        ).t().contiguous()  # (in_c,out_c) -> we need (out_c,in_c)

        merged = merged.view(self.base.out_channels, self.base.in_channels, 1, 1)
        self.base.weight += merged * self.scaling
        # zero out LoRA so it has no effect
        self.lora_A.weight.zero_()
        self.lora_B.weight.zero_()