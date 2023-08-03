import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision.datasets import StanfordCars
from torchvision.datasets.vision import VisionDataset


class StanfordCarsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        train_transform=None,
        test_transform=None,
        download: bool = False,
    ):
        super().__init__()

        self.train_dataset = StanfordCars(root, split="train", transform=train_transform, download=download)
        self.test_dataset = StanfordCars(root, split="test", transform=test_transform, download=download)

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
