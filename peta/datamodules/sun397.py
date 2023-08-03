import os
import re
from typing import List

import lightning.pytorch as pl
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


import os
import shutil
from pathlib import Path


def process_dataset(txt_file, downloaded_data_path, output_folder):
    R"""
    PROCESS SUN397 DATASET
    https://github.com/mlfoundations/task_vectors/issues/1#issuecomment-1514094158
    """
    with open(txt_file, "r") as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        input_path = line.strip()
        final_folder_name = "_".join(x for x in input_path.split("/")[:-1])[1:]
        filename = input_path.split("/")[-1]
        output_class_folder = os.path.join(output_folder, final_folder_name)

        if not os.path.exists(output_class_folder):
            os.makedirs(output_class_folder)

        full_input_path = os.path.join(downloaded_data_path, input_path[1:])
        output_file_path = os.path.join(output_class_folder, filename)
        # print(final_folder_name, filename, output_class_folder, full_input_path, output_file_path)
        # exit()
        shutil.copy(full_input_path, output_file_path)
        if i % 100 == 0:
            print(f"Processed {i}/{len(lines)} images")


if __name__ == "__main__":
    downloaded_data_path = "path/to/downloaded/SUN/data"
    process_dataset("Training_01.txt", downloaded_data_path, os.path.join(downloaded_data_path, "train"))
    process_dataset("Testing_01.txt", downloaded_data_path, os.path.join(downloaded_data_path, "val"))


class SUN397DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        train_transform=None,
        test_transform=None,
    ):
        super().__init__()

        self.train_dir = os.path.join(root, "sun397", "train")
        self.test_dir = os.path.join(root, "sun397", "test")

        self.train_transform = train_transform
        self.test_transform = test_transform

        self.loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=self.test_transform)

        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classes = [idx_to_class[i][2:].replace("_", " ") for i in range(len(idx_to_class))]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
