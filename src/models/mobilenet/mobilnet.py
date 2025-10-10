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
def create_model(num_class,cfg):
    if cfg.finetune =='lora':
        return wrap_mobilenet_v3_small_with_lora(num_classes=num_class,r= cfg.lora.r,alpha=cfg.lora.alpha,dropout=cfg.lora.dropout)
    if  cfg.finetune == 'head':
        model = HeadFTMobilNet(num_class=num_class)
    if  cfg.finetune== 'full':
        model = FullFTMobilNet(num_class=num_class)
    return model
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from src.optimizer.schedule_utils import create_cosine,create_warmup
from src.run_utils.lighting_utils import init,training_step,validation_step,configure_optimizers,test_step
class LitMobileNet(LightningModule):
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