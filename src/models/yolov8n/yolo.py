
from ultralytics import YOLO
import torch
from torch import nn
from torchvision import models, transforms, datasets
from pytorch_lightning import LightningModule

from src.fine_tune.lora_conv1 import LoRAConv2d


def wrap_1x1_convs_with_lora(root: nn.Module, r=8, alpha=16, dropout=0.0, skip_head=True):
    """
    Replace 1x1, groups=1 Conv2d with LoRAConv2d across the model.
    Optionally skip wrapping the Detect head (so its convs remain normal, trainable).
    """
    # optionally identify head subtree to skip
    head_sub = None
    if skip_head and hasattr(root, "model"):
        head_sub = detect_head_module(root)

    def _maybe_wrap(module: nn.Module, parent: nn.Module, name: str):
        child = getattr(parent, name)
        if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) and child.groups == 1:
            setattr(parent, name, LoRAConv2d(child, r=r, alpha=alpha, dropout=dropout))
        else:
            for sub_name, sub_child in list(child.named_children()):
                # Skip the head subtree if requested
                if skip_head and (head_sub is not None) and (sub_child is head_sub):
                    continue
                _maybe_wrap(sub_child, child, sub_name)

    for n, c in list(root.named_children()):
        # Skip the head as a whole if requested
        if skip_head and (head_sub is not None) and (c is head_sub):
            continue
        _maybe_wrap(c, root, n)

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

def merge_all_lora_(det_model: nn.Module):
    for m in det_model.modules():
        if isinstance(m, LoRAConv2d):
            m.merge_to_base_()

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
def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True

def load_yolo(pretrained = True):
    weights = 'yolov8s-cls.pt' if pretrained else 'yolov8s-cls.yaml'
    yolo = YOLO(weights)            
    model = yolo.model
    return model       
def detect_head_module(_):  # не нужен для cls, заглушка
    return None

import numpy as np
def make_class_weights(targets, num_classes, strategy="inverse", beta=0.999):
    counts = np.bincount(targets, minlength=num_classes)
    if strategy == "inverse":
        w = 1.0 / np.clip(counts, 1, None)
    elif strategy == "effective":
        w = (1.0 - beta) / (1.0 - np.power(beta, np.clip(counts, 1, None)))
    else:
        raise ValueError
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


def create_model(mode,num_class):
    if mode == 'head':
        return HeadYoloCls(num_class=num_class)
    if mode == 'lora':
        return  LoraYoloCls(num_class=num_class)

def replace_last_linear(model: nn.Module, out_features: int) -> nn.Linear:
    """
    Ищем ПОСЛЕДНИЙ nn.Linear в дереве и заменяем.
    Возвращаем ССЫЛКУ на новый слой, чтобы его можно было явно сохранить
    как self.classifier_head (для регистрации в Lightning).
    """
    last_parent, last_name, last_linear = None, None, None

    def dfs(parent: nn.Module):
        nonlocal last_parent, last_name, last_linear
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                last_parent, last_name, last_linear = parent, name, child
            dfs(child)

    dfs(model)
    if last_linear is None:
        raise RuntimeError("Не найден финальный nn.Linear в классификационной модели YOLOv8.")

    in_f = last_linear.in_features
    new_fc = nn.Linear(in_f, out_features)
    setattr(last_parent, last_name, new_fc)  # ВСТАВИЛИ В ДЕРЕВО
    return new_fc  # вернём ссылку
class LoraYoloCls(nn.Module):
    def __init__(self, num_class: int, pretrained: bool = True):
        super().__init__()
        self.model = load_yolo(pretrained=pretrained)

        # Replace the last Linear robustly
        self.classifier_head = replace_last_linear(self.model, num_class)

        # Wrap 1x1 convs with LoRA (backbone + neck for cls models)
        wrap_1x1_convs_with_lora(self.model, r=4, alpha=8, dropout=0, skip_head=True)

        # Freeze everything, then unfreeze LoRA params + classifier head
        freeze(self.model)
        # Unfreeze LoRA A/B
        for m in self.model.modules():
            if isinstance(m, LoRAConv2d):
                for p in m.lora_A.parameters(): p.requires_grad = True
                for p in m.lora_B.parameters(): p.requires_grad = True
        # Unfreeze classifier head as well
        unfreeze(self.classifier_head)

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)): out = out[0]
        if hasattr(out, "logits"): out = out.logits
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Expected Tensor logits, got {type(out)}")
        return out

class HeadYoloCls(nn.Module):
    def __init__(self, num_class: int, pretrained: bool = True):
        super().__init__()
        self.model = load_yolo(pretrained=pretrained)  # <- регистрируем как submodule
        freeze(self.model)                                  # всё фризим

        # заменяем последний Linear и храним явную ссылку
        self.classifier_head = replace_last_linear(self.model, num_class)
        unfreeze(self.classifier_head)                      # разморозим только голову

    def forward(self, x):
        out = self.model(x)
        # приведение к logits Tensor
        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "logits"):
            out = out.logits
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"Expected Tensor logits, got {type(out)}")
        return out
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from src.optimizer.schedule_utils import create_cosine,create_warmup
class LitYOLOCls(LightningModule):
    def __init__(self, cfg, num_class,class_weights = None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = create_model(cfg.finetune,num_class=num_class)
        self.loss = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
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
        opt = self.trainer.optimizers[0]
        lr0 = opt.param_groups[0]["lr"]
        self.log("lr", lr0, on_step=True, prog_bar=True, logger=True)
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
        max_epochs = int(self.cfg.trainer.max_epochs)
        warmup_epochs = max(1,int(0.05*max_epochs))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.trainer.lr)
        sched_warmup = create_warmup(warmup_epochs=warmup_epochs,optimizer=optimizer)
        sched_cosine = create_cosine(T_max= (max_epochs - warmup_epochs),optimizer=optimizer,eta_min=(self.cfg.trainer.lr * 0.001)) 

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[sched_warmup,sched_cosine],
            milestones=[warmup_epochs]
        )

        return {
            "optimizer":optimizer,
            "lr_scheduler":{
                "scheduler":scheduler,
                "interval":"epoch",
                "frequency":1
            }

        } 
