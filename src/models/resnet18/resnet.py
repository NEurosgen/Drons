import torch
import torch.nn as nn
from torchvision import models

# ---- Head FT: заморозка фичей + свой head ----
class HeadFTResNet18(nn.Module):
    def __init__(self, num_class: int):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        in_features = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_class),
        )
        # размораживаем только head
        for p in m.fc.parameters():
            p.requires_grad = True
        self.model = m

    def forward(self, x): return self.model(x)


# ---- Full FT: полностью обучаемая ----
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
