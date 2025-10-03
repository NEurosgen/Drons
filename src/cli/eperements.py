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
from run_exp import run_experiment
from src.sweep.sweeps import sweep_experiments
def set_seed(s):
    import random, numpy as np
    random.seed(s); torch.manual_seed(s); np.random.seed(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
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
    param_grid =  {
    'trainer.lr':[1e-3,1e-4],
    'name': ['resnet','mobilnet']
    }
    num_class = count_classes(cfg.path + '/train')
    sweep_experiments(cfg,param_grid=param_grid,run_single_trial=run_experiment,num_class=num_class,seeds = (42,21,11,15))
    #print(run_experiment(cfg=cfg,num_class=num_class,seeds = (42,21)))



if __name__ == "__main__":
    main()