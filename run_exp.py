import os
from pathlib import Path
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from src.models.create_model import create_model
from src.dataset_lit.ImageLoader import ImageLighting
# from src.models import create_model  # must return a LightningModule


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _to_scalar(x):
    """Turn PL/Torch metric values into Python floats/strs safely."""
    try:
        if hasattr(x, "item"):
            return float(x.item())
        if hasattr(x, "detach"):
            return float(x.detach().cpu().numpy())
        return float(x)
    except Exception:
        # as a fallback, stringify (for things like paths or non-floats)
        return str(x)
def _normalize_metric_keys(metrics_dict):
    """Map 'val/loss' -> 'val_loss', etc., and cast to floats."""
    out = {}
    for k, v in metrics_dict.items():
        k_norm = k.replace("/", "_")
        out[k_norm] = _to_scalar(v)
    return out

def exp_train(cfg,num_class ,seed: int):
    set_seed(seed)

    run_dir = Path(f"runs/{cfg.dataset}/seed_{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dm = ImageLighting(
        path_dir=cfg.path,
        batch_size=cfg.batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
    )

    model = create_model(cfg,num_class)
    print(f"\nNumber of parameters: {count_parameters(model):,}")

    # Save the best checkpoint by validation metric
    # monitor_metric = getattr(cfg.trainer, "monitor", "val_loss")
    # monitor_mode   = getattr(cfg.trainer, "monitor_mode", "min")
    # ckpt_cb = ModelCheckpoint(
    #     dirpath=str(run_dir),
    #     filename="{epoch}-{step}-{"+monitor_metric+":.4f}",
    #     monitor=monitor_metric,
    #     mode=monitor_mode,
    #     save_top_k=1
    # )

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        default_root_dir=str(run_dir),
        # callbacks=[ckpt_cb],
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=dm)
    cm = dict(trainer.callback_metrics)
    cm_norm = _normalize_metric_keys(cm)


    report = {}
    for name in ("val_loss", "val_acc", "val_f1", "hp_metric"):
        if name in cm_norm:
            report[name] = cm_norm[name]


    if not report:
        print("Report was empty. Available callback_metrics keys:",
              sorted(cm.keys()))

    return report


def run_experiment(cfg,num_class ,seeds):
    """
    Returns: list of tuples (seed, report_dict, best_ckpt_path_or_None)
    Example report_dict: {'val_loss': 0.3123, 'val_acc': 0.902, 'best_ckpt': '.../epoch=4-step=500.ckpt'}
    """
    results = []
    for seed in seeds:
        report = exp_train(cfg,num_class ,seed)
        results.append((seed, report))
    return results
