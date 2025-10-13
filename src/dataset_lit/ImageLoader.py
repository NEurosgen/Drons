from __future__ import annotations

from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


class ImageDS(Dataset):
    """Thin wrapper around :class:`torchvision.datasets.ImageFolder`."""

    def __init__(self, path_dir: Path | str, transform):
        super().__init__()
        self.ds = datasets.ImageFolder(path_dir, transform=transform)

    def __len__(self) -> int:  # pragma: no cover - simple passthrough
        return len(self.ds)

    def __getitem__(self, idx):  # pragma: no cover - simple passthrough
        return self.ds[idx]


class ImageLighting(LightningDataModule):
    """Lightning ``DataModule`` for image classification datasets."""

    def __init__(self, path_dir: Path | str, batch_size: int, train_transform, val_transform):
        super().__init__()
        self.path_dir = Path(path_dir)
        self.batch_size = batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform

        self.train_ds: Optional[ImageDS] = None
        self.val_ds: Optional[ImageDS] = None
        self.test_ds: Optional[ImageDS] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/validation/test datasets if needed."""

        if stage in (None, "fit", "validate"):
            self.train_ds = ImageDS(self.path_dir / "train", self.train_transform)
            self.val_ds = ImageDS(self.path_dir / "val", self.val_transform)

        if stage in (None, "fit", "test"):
            self.test_ds = ImageDS(self.path_dir / "test", self.val_transform)

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            raise RuntimeError("setup() must be called before requesting the train dataloader")
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            raise RuntimeError("setup() must be called before requesting the val dataloader")
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            raise RuntimeError("setup() must be called before requesting the test dataloader")
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
