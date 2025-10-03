import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics as tm
from torchvision import models
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
# ---------- ResNet18 backbones ----------
class HeadFTResNet18(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Tanh(),
            #nn.Dropout(0.2),
            nn.Linear(256, num_class),
        )
        # train only head
        for p in m.fc.parameters():
            p.requires_grad = True
        self.model = m
    def forward(self, x): return self.model(x)


class FullFTResNet18(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        m = models.resnet18(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_class),
        )
        self.model = m
    def forward(self, x): return self.model(x)


# ---- LoRA support (reuse your LoRAConv2d) ----
# from your code: class LoRAConv2d(...)

def wrap_resnet18_with_lora(num_classes: int, r=8, alpha=16, dropout=0.0, pretrained=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)

    # freeze all, then unfreeze fc + LoRA
    for p in m.parameters(): p.requires_grad = False
    for p in m.fc.parameters(): p.requires_grad = True

    def maybe_wrap(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d) and child.kernel_size == (1,1) and child.groups == 1:
                setattr(module, name, LoRAConv2d(child, r=r, alpha=alpha, dropout=dropout))
            else:
                maybe_wrap(child)

    maybe_wrap(m)  # wraps 1x1 convs (mainly in downsample paths)
    return m


# ---------- Fabric similar to your create_model ----------
def create_resnet18(name: str, num_class: int):
    if name == "head":
        return HeadFTResNet18(num_class)
    if name == "full":
        return FullFTResNet18(num_class)
    if name == "lora":
        return wrap_resnet18_with_lora(num_class)
    raise ValueError(f"Unknown resnet18 mode: {name}")


# ---------- LightningModule for ResNet18 ----------
class LitResNet18(LightningModule):
    def __init__(self, cfg, num_class: int):
        super().__init__()
        self.save_hyperparameters(ignore=["num_class"])
        self.cfg = cfg
        self.model = create_resnet18(cfg.finetune, num_class)
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)
        self.val_acc   = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)
        self.test_acc  = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc",  self.train_acc(preds, y), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc(preds, y), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc",  self.test_acc(preds, y), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        from torch.optim import AdamW
        return AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                     lr=self.cfg.trainer.lr)
