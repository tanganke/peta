# %%
import itertools
import logging
import os
from pathlib import Path
from typing import List

import hydra
import lightning as L
import lightning.pytorch as pl
import requests
import torch
import torch.nn.functional as F
import torchmetrics
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from peta.metrics.accuracy import Accuracy
from peta.models.clip import (
    CLIP_MODELS,
    freeze_unless_image_model,
    get_lora_vision_model,
    load_clip_model,
)
from peta.optim import CosineAnnealingWithWarmup
from peta.utils.logging import TitledLog, setup_colorlogging

log = logging.getLogger(__name__)


def setup_fabric(cfg: DictConfig):
    from lightning.fabric.loggers.tensorboard import TensorBoardLogger

    logger = TensorBoardLogger(
        root_dir=Path("logs") / cfg.model_name / cfg.dataset_name,
        name=cfg.finetuning_mode,
    )
    fabric = L.Fabric(**cfg.fabric, loggers=logger)
    fabric.launch()
    return fabric


def train(
    fabric: L.Fabric,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    train_loader: DataLoader,
    num_classes: int,
    text: List[str],
    max_steps: int = 1,
    num_grad_accumulation: int = 1,
):
    epoch_idx = -1
    clip_model.train()
    with TitledLog(" Train ", log_fn=log.info):
        for step_idx, (batch_idx, batch) in enumerate(
            itertools.cycle(
                enumerate(train_loader),
            )
        ):
            if batch_idx == 0:
                epoch_idx += 1

            is_accumalating = step_idx % num_grad_accumulation
            images, labels = batch
            with fabric.no_backward_sync(clip_model, enabled=is_accumalating):
                inputs = processor(
                    text=text, images=images, return_tensors="pt", padding=True
                )
                # send tensors to device
                for k in inputs:
                    if isinstance(inputs[k], Tensor):
                        inputs[k] = inputs[k].to(fabric.device)

                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                preds = logits_per_image.argmax(dim=-1)

                loss = F.cross_entropy(logits_per_image, labels)
                fabric.log("train/loss", loss.item(), step_idx)

                fabric.backward(loss)

                print(
                    f"[Training: epoch={epoch_idx},batch={batch_idx}] step: {step_idx} loss: {loss.item():.3f}"
                )

            if not is_accumalating:
                fabric.clip_gradients(clip_model, optimizer, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step(step_idx)

                if step_idx >= max_steps * num_grad_accumulation:
                    # terminate training
                    break


def test(
    fabric: L.Fabric,
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    test_loader: DataLoader,
    num_classes: int,
    step_idx: int,
    text: List[str],
):
    # metrics
    accuracy = torchmetrics.Accuracy(
        "multiclass",
        num_classes=num_classes,
    ).to(fabric.device)

    clip_model.eval()
    with TitledLog(" Test ", log_fn=log.info):
        for batch_idx, batch in enumerate(tqdm(test_loader, "test")):
            images, labels = batch
            inputs = processor(
                text=text, images=images, return_tensors="pt", padding=True
            )
            # send tensors to device
            for k in inputs:
                if isinstance(inputs[k], Tensor):
                    inputs[k] = inputs[k].to(fabric.device)

            outputs = clip_model(**inputs)
            logits_per_image: Tensor = outputs.logits_per_image
            preds = logits_per_image.argmax(dim=-1)

            # metric
            accuracy(preds, labels)

    fabric.log("test/acc", accuracy.compute().item(), step_idx)


# %%
@hydra.main(
    config_path="config",
    config_name="finetune_clip",
    version_base=None,
)
def main(cfg: DictConfig):
    setup_colorlogging(force=True)

    fabric = setup_fabric(cfg)
    if fabric.logger is not None:
        # save `cfg` to fabric.logger.log_dir/config.yaml
        if not os.path.exists(fabric.logger.log_dir):
            os.makedirs(fabric.logger.log_dir)
        config_path = os.path.join(fabric.logger.log_dir, "config.yaml")
        OmegaConf.save(cfg, config_path)

    # create model
    with TitledLog(" Create model ", log_fn=log.info):
        model_name_or_path = cfg.model.model_name_or_path
        assert (
            model_name_or_path in CLIP_MODELS
        ), f"Unknown model name or path: {model_name_or_path}"
        processor, clip_model = load_clip_model(model_name_or_path)
        clip_model = freeze_unless_image_model(clip_model)

        if cfg.lora_config is not None:
            lora_config = instantiate(cfg.lora_config)
            lora_vision_model = get_peft_model(clip_model.vision_model, lora_config)
            lora_vision_model.print_trainable_parameters()
            clip_model.vision_model = lora_vision_model

        # setup optimizer
        optimizer = torch.optim.AdamW(
            [p for p in clip_model.vision_model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        lr_scheduler = CosineAnnealingWithWarmup(
            optimizer,
            base_lrs=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            max_steps=cfg.max_steps,
        )
        clip_model = fabric.setup_module(clip_model)

    # load data
    with TitledLog(" Load data ", log_fn=log.info):
        assert (
            cfg.model.batch_size % cfg.fabric.devices == 0
        ), "batch_size must be divisible by devices"
        cfg.batch_size = cfg.model.batch_size // cfg.fabric.devices
        input_size = cfg.model.input_size
        datamodule: pl.LightningDataModule = instantiate(
            cfg.datamodule,
            train_transform=transforms.Compose(
                [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
            ),
            test_transform=transforms.Compose(
                [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
            ),
        )
        train_loader = datamodule.train_dataloader()
        test_loader = datamodule.test_dataloader()
        print("training dataset", train_loader.dataset)
        print("test dataset", test_loader.dataset)

        train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    test(
        fabric=fabric,
        clip_model=clip_model,
        processor=processor,
        test_loader=test_loader,
        num_classes=len(datamodule.classes),
        step_idx=0,
        text=[f"a photo of a {c}" for c in datamodule.classes],
    )

    # train
    train(
        fabric=fabric,
        clip_model=clip_model,
        processor=processor,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        num_classes=len(datamodule.classes),
        text=[f"a photo of a {c}" for c in datamodule.classes],
        max_steps=cfg.max_steps,
        num_grad_accumulation=cfg.num_grad_accumulation,
    )

    test(
        fabric=fabric,
        clip_model=clip_model,
        processor=processor,
        test_loader=test_loader,
        num_classes=len(datamodule.classes),
        step_idx=cfg.max_steps,
        text=[f"a photo of a {c}" for c in datamodule.classes],
    )


if __name__ == "__main__":
    main()
