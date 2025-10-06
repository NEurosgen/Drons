# # yolo_v8_baseline.py
# import warnings
# warnings.filterwarnings("ignore")

# import torch
# import torch.nn as nn

# # pip install ultralytics==8.3.0
from ultralytics import YOLO
import torch
from torch import nn
from torchvision import models, transforms, datasets
from pytorch_lightning import LightningModule
# # ------------------------------
# # LoRA for 1x1 Conv2d (groups=1)
# # ------------------------------
# class LoRAConv2d(nn.Module):
#     """
#     LoRA wrapper for 1x1 Conv2d (groups=1).
#     Output = base_conv(x) + (alpha/r) * B(A(x))
#     Base conv is frozen; only A/B are trainable.
#     """
#     def __init__(self, base_conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0):
#         super().__init__()
#         assert isinstance(base_conv, nn.Conv2d)
#         assert base_conv.kernel_size == (1, 1) and base_conv.groups == 1, \
#             "LoRAConv2d supports only 1x1 groups=1 convs."

#         self.base = base_conv
#         for p in self.base.parameters():
#             p.requires_grad = False

#         self.r = r
#         self.alpha = alpha
#         self.scaling = alpha / r

#         in_c  = base_conv.in_channels
#         out_c = base_conv.out_channels

#         self.lora_A = nn.Conv2d(in_c,  r,    kernel_size=1, bias=False)
#         self.lora_B = nn.Conv2d(r,     out_c, kernel_size=1, bias=False)

#         nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
#         nn.init.zeros_(self.lora_B.weight)

#         self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

#     def forward(self, x):
#         base = self.base(x)
#         delta = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
#         return base + delta

#     @torch.no_grad()
#     def merge_to_base_(self):
#         # Merge B@A into base weight (in-place)
#         merged = torch.matmul(
#             self.lora_B.weight.view(self.lora_B.out_channels, self.lora_B.in_channels),
#             self.lora_A.weight.view(self.lora_A.in_channels, self.lora_A.out_channels)
#         ).t().contiguous()  # -> (out_c, in_c)
#         merged = merged.view(self.base.out_channels, self.base.in_channels, 1, 1)
#         self.base.weight += merged * self.scaling
#         self.lora_A.weight.zero_()
#         self.lora_B.weight.zero_()

# # ------------------------------
# # Utilities to control trainable
# # ------------------------------
# def freeze_all(m: nn.Module):
#     for p in m.parameters():
#         p.requires_grad = False

# def unfreeze(m: nn.Module):
#     for p in m.parameters():
#         p.requires_grad = True



# def wrap_1x1_convs_with_lora(root: nn.Module, r=8, alpha=16, dropout=0.0, skip_head=True):
#     """
#     Replace 1x1, groups=1 Conv2d with LoRAConv2d across the model.
#     Optionally skip wrapping the Detect head (so its convs remain normal, trainable).
#     """
#     # optionally identify head subtree to skip
#     head_sub = None
#     if skip_head and hasattr(root, "model"):
#         head_sub = detect_head_module(root)

#     def _maybe_wrap(module: nn.Module, parent: nn.Module, name: str):
#         child = getattr(parent, name)
#         if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) and child.groups == 1:
#             setattr(parent, name, LoRAConv2d(child, r=r, alpha=alpha, dropout=dropout))
#         else:
#             for sub_name, sub_child in list(child.named_children()):
#                 # Skip the head subtree if requested
#                 if skip_head and (head_sub is not None) and (sub_child is head_sub):
#                     continue
#                 _maybe_wrap(sub_child, child, sub_name)

#     for n, c in list(root.named_children()):
#         # Skip the head as a whole if requested
#         if skip_head and (head_sub is not None) and (c is head_sub):
#             continue
#         _maybe_wrap(c, root, n)

# def set_trainable_mode(det_model: nn.Module, mode: str):
#     """
#     mode in {"head", "full", "lora"}
#     - head: freeze backbone+neck, train only detection head
#     - full: train all
#     - lora: freeze backbone+neck base weights, train LoRA params + head
#     """
#     head = detect_head_module(det_model)

#     if mode == "full":
#         unfreeze(det_model)

#     elif mode == "head":
#         freeze_all(det_model)
#         unfreeze(head)

#     elif mode == "lora":
#         # Freeze everything
#         freeze_all(det_model)
#         # Unfreeze LoRA params (A/B) and head
#         for m in det_model.modules():
#             if isinstance(m, LoRAConv2d):
#                 for p in m.lora_A.parameters():
#                     p.requires_grad = True
#                 for p in m.lora_B.parameters():
#                     p.requires_grad = True
#         unfreeze(head)

#     else:
#         raise ValueError(f"Unknown finetune mode: {mode}")

# def merge_all_lora_(det_model: nn.Module):
#     for m in det_model.modules():
#         if isinstance(m, LoRAConv2d):
#             m.merge_to_base_()

# # ------------------------------
# # Factory similar to your create_model()
# # ------------------------------
# def create_yolov8s_detector(
#     finetune: str = "head",   # "head" | "full" | "lora"
#     num_classes: int | None = None,
#     lora_r: int = 8,
#     lora_alpha: int = 16,
#     lora_dropout: float = 0.0,
#     pretrained_weights: str = "yolov8s.pt",   # or a custom .pt
# ):
#     """
#     Returns an Ultralytics YOLO object with underlying nn.Module prepared for finetuning.
#     """
#     yolo = YOLO(pretrained_weights)  # loads model+weights
#     det = yolo.model                 # underlying torch model

#     # Optionally change number of classes (if different from pretrain)
#     if (num_classes is not None) and hasattr(det, "nc") and det.nc != num_classes:
#         det.nc = num_classes
#         # Reinit Detect head anchors/cls channels automatically via Ultralytics:
#         # yolo.model = det  # not needed; it's the same reference
#         # NOTE: Ultralytics will adjust head on .train() when data YAML is provided.
#         # If you want to hard-override head channels now, you can rebuild Detect head,
#         # but letting Ultralytics handle it via data YAML is usually easier.

#     if finetune == "lora":
#         # Wrap 1x1 convs (backbone+neck) with LoRA; keep head normal
#         wrap_1x1_convs_with_lora(det, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, skip_head=True)

#     # Set which params are trainable
#     set_trainable_mode(det, finetune)

#     return yol


import torch
import torch.nn as nn
from ultralytics import YOLO
from pytorch_lightning import LightningModule
import torchmetrics as tm

def detect_head_module(_):  # не нужен для cls, заглушка
    return None

class HeadYoloCls(nn.Module):
    def __init__(self, num_class, pretrained=True):
        super().__init__()
        weights = 'yolov8s-cls.pt' if pretrained else 'yolov8s-cls.yaml'
        yolo = YOLO(weights)            
        self.model = yolo.model              
        for p in self.model.parameters():
            p.requires_grad = False

        in_f = self.model.model[9].linear.in_features
        self.model.model[9].linear = nn.Linear(in_f, num_class)
        for p in self.model.model[9].parameters():
            p.requires_grad = True

    def forward(self, x):
        out = self.model(x)
        # Ultralytics иногда возвращает list/tuple/obj
        if isinstance(out, (list, tuple)):
            out = out[0]
        # Некоторые версии могут вернуть объект с .logits
        if hasattr(out, "logits"):
            out = out.logits
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Expected Tensor logits, got {type(out)}")
        return out

class LitYOLOCls(LightningModule):
    def __init__(self, cfg, num_class):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = HeadYoloCls(num_class=num_class, pretrained=True)
        self.loss = nn.CrossEntropyLoss()
        self.cfg = cfg
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)
        self.val_acc   = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)
        self.test_acc  = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)              # [B, C]
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=1)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_acc.update(preds, y)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=1)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_acc.update(preds, y)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        preds = logits.argmax(dim=1)
        self.log("test_loss", loss, on_epoch=True)
        self.test_acc.update(preds, y)
        self.log("test_acc", self.test_acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.trainer.lr)
