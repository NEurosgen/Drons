
import torch

from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms
from pytorch_lightning import LightningDataModule
def data_stats(data_dir,transform):

    ds = datasets.ImageFolder(data_dir,transform = transform)
    batch_size = 32
    dl = DataLoader(ds,batch_size=batch_size,num_workers=0,pin_memory=True)

    n_pixels = 0
    mean = torch.zeros(3)
    M2 = torch.zeros(3)

    for x,_  in dl:
        B,C,H,W = x.shape
        x = x.view(B,C,-1)
        batch_pixel = B*H*W

        b_mean = x.mean(dim = (0,2))
        b_var = x.var(dim=(0,2),unbiased=False)

        delta = b_mean - mean
        total= n_pixels + batch_pixel
        new_mean = mean+delta*(batch_pixel/total) 
        M2 = M2 + b_var*batch_pixel+ (delta**2)*(n_pixels*batch_pixel/total)
        mean = new_mean
        n_pixels = total

    var = M2/n_pixels
    std = torch.sqrt(var)
    return mean,std

class ImageDS(Dataset):
    def __init__(self,path_dir,tranform):
        super().__init__()
        self.ds = datasets.ImageFolder(path_dir,transform=tranform)

    def __len__(self):
        return len(self.ds)
    def __getitem__(self,idx):
        return self.ds[idx]
    
class ImageLighting(LightningDataModule):
    def __init__(self,path_dir,batch_size,train_transform,val_transform):
        super().__init__()
        self.path_dir = path_dir
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform


    def setup(self, stage):
        self.train_ds,self.val_ds,self.test_ds =   ImageDS(self.path_dir+'/train',self.train_transform),ImageDS(self.path_dir+'/val',self.val_transform),ImageDS(self.path_dir+'/test',self.val_transform)
           
    def train_dataloader(self):
        return DataLoader(self.train_ds,batch_size=self.batch_size,shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_ds,batch_size=self.batch_size,shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test_ds,batch_size=self.batch_size,shuffle=False)