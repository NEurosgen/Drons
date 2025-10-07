import torch
from torch import nn
from torchvision import models, transforms, datasets
from pytorch_lightning import LightningModule
import torchmetrics as tm
class HeadFTMobilNet(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.model =  models.mobilenet_v3_small(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        feat_num = self.model.classifier[-1].in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(576, 256),  
            nn.Hardswish(), 
            nn.Dropout(0.2), #Maybe, I don't think so
            nn.Linear(256, num_class)
        )
    def forward(self,x):
        out = self.model(x)
        return out


class FullFTMobilNet(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.model =  models.mobilenet_v3_small(pretrained=False)
        feat_num = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(feat_num, 256),  
            nn.Hardswish(), 
           # nn.Dropout(0.2), Maybe, I don't think so
            nn.Linear(256, num_class)
        )
    def forward(self,x):
        out = self.model(x)
        return out

from src.fine_tune.lora_conv1 import LoRAConv2d

from torchvision import models

from torchvision import models
import torch.nn as nn

def wrap_mobilenet_v3_small_with_lora(
    num_classes: int,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    pretrained: bool = True
):

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    )

    in_features = model.classifier[3].in_features

    # Заменяем последний слой на наш
    model.classifier[3] = nn.Linear(in_features, num_classes)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    def maybe_wrap(m):
        for name, child in list(m.named_children()):
            if isinstance(child, nn.Conv2d) and child.kernel_size == (1, 1) and child.groups == 1:
                setattr(m, name, LoRAConv2d(child, r=r, alpha=alpha, dropout=dropout))
            else:
                maybe_wrap(child)

    maybe_wrap(model.features)

    return model




import hydra
def create_model(name,num_class):
    if name =='lora':
        return wrap_mobilenet_v3_small_with_lora(num_classes=num_class)
    if name == 'head':
        model = HeadFTMobilNet(num_class=num_class)
    if name == 'full':
        model = FullFTMobilNet(num_class=num_class)
    return model
from torch.optim import Adam
class LitMobileNet(LightningModule):
    def __init__(self,cfg,num_class):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = create_model(cfg.finetune,num_class)
        self.loss = nn.CrossEntropyLoss()
        self.cfg = cfg
        self.train_acc = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)
        self.val_acc   = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)
        self.test_acc  = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)

    def forward(self,batch):
        logits = self.model(batch)
        return logits

    def training_step(self, batch,batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        preds = logits.argmax(dim=1)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_acc.update(preds, y)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch,batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        preds = logits.argmax(dim=1)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_acc.update(preds, y)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

        return loss
    def test_step(self, batch,batch_idx):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        preds = logits.argmax(dim=1)

        self.log("test_loss", loss, on_epoch=True)
        self.test_acc.update(preds, y)
        self.log("test_acc", self.test_acc, on_epoch=True)
        return loss
    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(),lr=self.cfg.trainer.lr)

        return optimizer