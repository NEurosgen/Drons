import os, math, yaml, torch
from torch import optim
from tqdm import trange
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models, transforms, datasets
from src.dataset_lit.ImageLoader import ImageLighting
from src.models.mobilenet.mobilnet import LitMobileNet
from pathlib import Path
from src.models.create_model import create_model
from omegaconf import OmegaConf
def set_seed(s):
    import random, numpy as np
    random.seed(s); torch.manual_seed(s); np.random.seed(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
import numpy as np
import torch

def make_class_weights_from_folder(train_root: str, strategy: str = "effective", beta: float = 0.999) -> torch.Tensor:
    """
    Считает веса классов из ImageFolder(train_root).
    strategy: 'inverse' (1/n_c) или 'effective' (Cui et al. 2019)
    """
    # Без аугментаций: только привести к RGB, чтобы загрузчик не падал
    ds_tmp = datasets.ImageFolder(train_root, transform=transforms.Lambda(lambda im: im.convert("RGB")))
    targets = np.array(ds_tmp.targets)
    num_classes = len(ds_tmp.classes)

    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1, None)  # защита от деления на ноль

    if strategy == "inverse":
        w = 1.0 / counts
    elif strategy == "effective":
        # Class-Balanced Loss: w_c = (1 - beta) / (1 - beta**n_c)
        w = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Нормировка, чтобы средний вес ≈ 1
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)

def count_classes(path: str) -> int | None:
    """
    Counting number of classes in dataset.

    :param str path: Path to dataset.
    :returns int: Number of classes or None in case of an error.
    """
    try:
        entries = os.listdir(path)
        dir_count = sum(1 for entry in entries if os.path.isdir(os.path.join(path, entry)))
        return dir_count
    except Exception as e:
        print(f"ERROR: {e}")
        return None
CFG_DIR = Path(__file__).resolve().parents[2] / "configs" 
@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    train_root = os.path.join(cfg.path, 'train')
    class_weights = make_class_weights_from_folder(
        train_root,
        strategy=getattr(cfg, "imbalance_strategy", "effective"),  # можно положить в конфиг
        beta=getattr(cfg, "imbalance_beta", 0.999)
    )
    print(class_weights)
    device = torch.device(cfg["trainer"]["device"] if torch.cuda.is_available() else "cpu")
    logger = TensorBoardLogger(save_dir='./tb_logs',
                               name=cfg.name)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    train_transform =  transforms.Compose([transforms.RandomRotation(degrees=30),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.Resize((256,256)),
                                       transforms.CenterCrop((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)
                                       ])
    val_transform = transforms.Compose([transforms.Resize((256,256)),
                                     transforms.CenterCrop((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=mean, std=std)
                                     ])
    
    dm = ImageLighting(path_dir = cfg.path,batch_size=cfg.batch_size,train_transform=train_transform,val_transform = val_transform)
    num_class = count_classes(cfg.path+'/train')
    model = create_model(cfg,num_class=num_class,class_weights=class_weights)
    # checkpoint_callback = ModelCheckpoint(
    #     monitor = cfg.callbacks.model_checkpoint.monitor,
    #     mode = cfg.callbacks.model_checkpoint.mode,
    #     save_top_k=1,
    #     save_last=True,
    #     filename=f"best"
    # )
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else 'cpu',
        logger=logger,
        deterministic=True
    )
    trainer.fit(model,datamodule=dm)

if __name__ == "__main__":
    main()