from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from src.optimizer.schedule_utils import create_cosine,create_warmup

import torch
from torch import nn
from torchvision import models, transforms, datasets
from pytorch_lightning import LightningModule
import torchmetrics as tm

def init(module,cfg,num_class,create_model,class_weights = None):
    module.save_hyperparameters(cfg)
    module.model = create_model(num_class_ = num_class,cfg_ = cfg)
    if class_weights is not None:
        module.loss = nn.CrossEntropyLoss(weight=class_weights.to(module.device),label_smoothing=0.1)
    else:
        module.loss = nn.CrossEntropyLoss()
    module.cfg = cfg
    module.train_acc = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)
    module.val_acc   = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)
    module.test_acc  = tm.Accuracy(task="multiclass", num_classes=num_class, top_k=1)

def training_step(self,batch):
        x,y = batch
        logits = self(x)
        loss = self.loss(logits,y)
        preds = logits.argmax(dim=1)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_acc.update(preds, y)
        self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
        opt = self.trainer.optimizers[0]
        lr0 = opt.param_groups[0]["lr"]
        self.log("lr", lr0, on_step=True, prog_bar=True, logger=True)
        return loss
def validation_step(self, batch):
    x,y = batch
    logits = self(x)
    loss = self.loss(logits,y)
    preds = logits.argmax(dim=1)

    self.log("val_loss", loss, on_epoch=True, prog_bar=True)
    self.val_acc.update(preds, y)
    self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)

    return loss

def test_step(self, batch):
    x,y = batch
    logits = self(x)
    loss = self.loss(logits,y)
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
    sched_cosine = create_cosine(T_max= (max_epochs - warmup_epochs),optimizer=optimizer,eta_min=(self.cfg.trainer.lr * 0.0001)) 

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
