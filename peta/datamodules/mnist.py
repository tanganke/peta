import os

import lightning.pytorch as pl
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class MNISTDataModule(pl.LightningDataModule):
    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        download: bool = False,
        pin_memory: bool = True,
        train_transform=None,
        test_transform=None,
    ):
        super().__init__()
        self.root = root
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        self.train_dataset = datasets.MNIST(self.root, train=True, transform=self.train_transform, download=download)
        self.test_dataset = datasets.MNIST(self.root, train=False, transform=self.test_transform, download=download)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
