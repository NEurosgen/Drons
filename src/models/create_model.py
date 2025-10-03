from src.models.mobilenet.mobilnet import LitMobileNet
from src.models.resnet18.resnet import LitResNet18
def create_model(cfg,num_class):
    if cfg.name == 'resnet':
        return LitResNet18(cfg,num_class=num_class)
    if cfg.name =='mobilnet':
        return LitMobileNet(cfg,num_class=num_class)