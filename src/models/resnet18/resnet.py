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

# ---------- LightningModule for ResNet18 ----------
class LitResNet18(LightningModule):
    def __init__(self, cfg, num_class: int,class_weights = None):
        super().__init__()
        self.save_hyperparameters(ignore=["num_class"])
        self.cfg = cfg
        self.model = create_resnet18(cfg.finetune, num_class)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

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
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc",  self.train_acc(preds, y), on_step=True, on_epoch=True, prog_bar=True)
        opt = self.trainer.optimizers[0]
        lr0 = opt.param_groups[0]["lr"]
        self.log("lr", lr0, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc",  self.val_acc(preds, y), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc",  self.test_acc(preds, y), on_step=False, on_epoch=True)

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
