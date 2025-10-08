import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics as tm
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.fine_tune.lora_conv1 import LoRAConv2d

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
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from src.optimizer.schedule_utils import create_cosine,create_warmup
from src.run_utils.lighting_utils import init,training_step,validation_step,configure_optimizers,test_step
# ---------- LightningModule for ResNet18 ----------
class LitResNet18(LightningModule):
    def __init__(self, cfg, num_class: int,class_weights = None):
        super().__init__()

        init(self,cfg,num_class=num_class,create_model=create_resnet18,class_weights=class_weights)
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