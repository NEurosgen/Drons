
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


def merge_all_lora_(det_model: nn.Module):
    for m in det_model.modules():
        if isinstance(m, LoRAConv2d):
            m.merge_to_base_()



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


def create_model(num_class,cfg):
    if cfg.finetune == 'head':
        return HeadYoloCls(num_class=num_class)
    if cfg.finetune == 'lora':
        return  LoraYoloCls(num_class=num_class)

def replace_last_linear(model: nn.Module, out_features: int) -> nn.Linear:
    """
    Ищем ПОСЛЕДНИЙ nn.Linear в дереве и заменяем.
    Возвращаем ССЫЛКУ на новый слой, чтобы его можно было явно сохранить
    как self.classifier_head (для регистрации в Lightning).
    """
    # last_parent, last_name, last_linear = None, None, None

    # def dfs(parent: nn.Module):
    #     nonlocal last_parent, last_name, last_linear
    #     for name, child in parent.named_children():
    #         if isinstance(child, nn.Linear):
    #             last_parent, last_name, last_linear = parent, name, child
    #         dfs(child)

    # dfs(model)
    # if last_linear is None:
    #     raise RuntimeError("Не найден финальный nn.Linear в классификационной модели YOLOv8.")
    last_linear = model.model[-1].linear
    in_f = last_linear.in_features
    model.model[-1].linear = nn.Linear(in_f, out_features)
    return model.model[-1]  # вернём ссылку
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
        unfreeze(self.classifier_head.linear)

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
from src.run_utils.lighting_utils import init,training_step,validation_step,configure_optimizers,test_step
class LitYOLOCls(LightningModule):
    def __init__(self,cfg,num_class,class_weights = None):
        super().__init__()
        init(self,cfg,num_class=num_class,create_model=create_model,class_weights=class_weights)
    def forward(self,batch):
        logits = self.model(batch)
        return logits

    def training_step(self, batch,batch_idx):
        loss = training_step(self,batch)
        return loss
    def validation_step(self, batch,batch_idx):
        loss = validation_step(self,batch=batch)
        return loss
    def test_step(self, batch,batch_idx):
        loss = test_step(self,batch)
    def configure_optimizers(self):
        report = configure_optimizers(self)

        return  report