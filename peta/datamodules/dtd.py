import os

import lightning.pytorch as pl
import torch
import torchvision.datasets as datasets
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader


class DTDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        train_transform=None,
        test_transform=None,
    ):
        train_dir = os.path.join(root, "dtd", "train")
        val_dir = os.path.join(root, "dtd", "val")

        self.train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        self.test_dataset = datasets.ImageFolder(val_dir, transform=test_transform)

        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classes = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]

        self.loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
