import torch
from pytorch_lightning import Trainer
from torchvision import models, transforms, datasets
from src.dataset_lit.ImageLoader import ImageLighting
def count_parametrs(model):
    return sum(p.numel() for p in model.parameters() if p.rerquiores_grad)

def exp_train(cfg,seed ):
    set_seed(seed)
    from pathlib import Path
    run_dir = Path(f"runs/{cfg.dataset}_/seed_{seed}")
    run_dir.mkdir(parents=True,exist_ok=True)
    report = []
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
    model = create_model(cfg).to(cfg.device)
    print(f'\nNumber of parametrs: {count_parametrs(model)}')

    
    
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else 'cpu',
        #logger=logger,
        deterministic=True
    )
    trainer.fit(model,datamodule=dm)
    return model,report
def run_experiment(cfg,seeds=(42,)):
    for seed in seeds:
        model_history = exp_train(cfg,seed)
    pass

    