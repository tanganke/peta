import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List

import lightning as L
import lightning.pytorch as pl
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from finetune_clip import load_clip_model, load_clip_processor_and_model
from peta.tasks.arithmetic import state_dict_add, state_dict_sub
from peta.utils import TitledLog

log = logging.getLogger(__name__)

MODEL_NAME = "ViT-B-16"
MODEL_NAME_OR_PATH = "openai/clip-vit-base-patch16"
VERSION = 0
STEPS = 2000
DATASET_NAMES = ["Cars", "DTD", "EuroSAT", "GTSRB", "RESISC45", "SUN397", "SVHN"]


def evaluate_accuracy(
    *,
    # model
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    # data
    text: List[str],
    test_loader: DataLoader,
) -> float:
    clip_model.eval()
    # precompute the text features
    text_input = clip_processor(text, return_tensors="pt", padding=True)
    text_embeds = clip_model.get_text_features(**text_input)

    correct, count = 0, 0
    with TitledLog("Evaluate accuracy", log_fn=log.info):
        for batch in tqdm(test_loader):
            images, labels = batch
            with torch.no_grad():
                image_embeds = clip_model.get_image_features(pixel_values=images)

                # normalized features
                image_embeds = image_embeds / image_embeds.norm(
                    p=2, dim=-1, keepdim=True
                )
                text_embeds = text_embeds.to(image_embeds.device)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

                # cosine similarity as logits
                logit_scale = clip_model.logit_scale.exp().item()
                logits_per_text = (
                    torch.matmul(text_embeds, image_embeds.t()) * logit_scale
                )
                logits_per_image = logits_per_text.t()

                pred = logits_per_image.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                count += len(labels)
    return correct / count


pretrained_clip_vision_models = {}
finetuned_clip_vison_models_task_vectors = {}
datamodules = {}
train_loaders = {}
test_loaders = {}


def load_models_and_datasets():
    for dataset_name in DATASET_NAMES:
        for finetune_mode in ["standard", "lora", "l_lora"]:
            log_dir = (
                Path("logs")
                / MODEL_NAME
                / dataset_name
                / finetune_mode
                / f"version_{VERSION}"
            )
            if not log_dir.exists():
                log.warning(f"skip {log_dir}")
                continue
            log.info(f"load {log_dir}")
            cfg = OmegaConf.load(log_dir / "config.yaml")

            if dataset_name not in datamodules:
                log.info(f"load dataset {dataset_name}")
                with TitledLog(" Load data ", log_fn=log.info):
                    assert (
                        cfg.model.batch_size % cfg.fabric.devices == 0
                    ), "batch_size must be divisible by devices"
                    cfg.batch_size = cfg.model.batch_size // cfg.fabric.devices
                    input_size = cfg.model.input_size
                    datamodule: pl.LightningDataModule = instantiate(
                        cfg.datamodule,
                        train_transform=transforms.Compose(
                            [
                                transforms.Resize((input_size, input_size)),
                                transforms.ToTensor(),
                            ]
                        ),
                        test_transform=transforms.Compose(
                            [
                                transforms.Resize((input_size, input_size)),
                                transforms.ToTensor(),
                            ]
                        ),
                    )
                    train_loader = datamodule.train_dataloader()
                    test_loader = datamodule.test_dataloader()
                    print("training dataset", train_loader.dataset)
                    print("test dataset", test_loader.dataset)

                    datamodules[dataset_name] = datamodule
                    train_loaders[dataset_name] = train_loader
                    test_loaders[dataset_name] = test_loader

            # load model
            if finetune_mode not in pretrained_clip_vision_models:
                log.info(f"load pre-trained model for {finetune_mode} - {dataset_name}")
                (
                    clip_processor,
                    clip_model,
                    clip_vision_model,
                    clip_text_model,
                ) = load_clip_processor_and_model(
                    cfg.model.model_name_or_path,
                    cfg.lora_config,
                    linearized_lora=cfg.linearized_lora,
                    random_seed=cfg.seed,
                )

                pretrained_clip_vision_models[finetune_mode] = clip_vision_model

            # load checkpoints
            if finetune_mode not in finetuned_clip_vison_models_task_vectors:
                finetuned_clip_vison_models_task_vectors[finetune_mode] = {}
            if (
                dataset_name
                not in finetuned_clip_vison_models_task_vectors[finetune_mode]
            ):
                log.info(f"load task vector for {finetune_mode} - {dataset_name}")
                pretrained_model = pretrained_clip_vision_models[finetune_mode]
                ckpt_path = log_dir / "checkpoints" / f"vision_model-step={STEPS}.pth"
                state_dict = torch.load(ckpt_path, map_location="cpu")
                state_dict = {
                    (".".join(k.split(".")[1:])): p for k, p in state_dict.items()
                }
                assert set(state_dict.keys()).issubset(
                    pretrained_model.state_dict().keys()
                )
                finetuned_clip_vison_models_task_vectors[finetune_mode][
                    dataset_name
                ] = state_dict_sub(
                    state_dict, pretrained_model.state_dict(), strict=False
                )


load_models_and_datasets()

if __name__ == "__main__":
    fabric = L.Fabric(accelerator="gpu", devices=1)
    fabric.launch()

    results = defaultdict(lambda: list())

    for dataset_name in DATASET_NAMES:
        for finetune_mode in ["standard", "lora", "l_lora"]:
            log.info(
                f"evaluate zero shot accuracy for {finetune_mode} - {dataset_name}"
            )
            # evaluate zero shot accuracy
            clip_processor, clip_model = load_clip_model(
                MODEL_NAME_OR_PATH, local_files_only=True
            )
            clip_model.vision_model = deepcopy(
                pretrained_clip_vision_models[finetune_mode]
            )
            # setup fabric modules
            clip_model.vision_model = fabric.setup_module(clip_model.vision_model)
            clip_model.visual_projection = fabric.setup_module(
                clip_model.visual_projection
            )

            datamodule = datamodules[dataset_name]
            test_loader = fabric.setup_dataloaders(test_loaders[dataset_name])
            text = [f"a photo of a {c}" for c in datamodule.classes]

            acc = evaluate_accuracy(
                clip_model=clip_model,
                clip_processor=clip_processor,
                text=text,
                test_loader=test_loader,
            )

            results["task"].append(dataset_name)
            results["finetune_mode"].append(finetune_mode)
            results["step"].append(0)
            results["accuracy"].append(acc)

            print(pd.DataFrame(results))

        for finetune_mode in ["standard", "lora", "l_lora"]:
            # evaluate finetuned accuracy
            log.info(
                f"evaluate finetuned accuracy for {finetune_mode} - {dataset_name}"
            )
            clip_processor, clip_model = load_clip_model(
                MODEL_NAME_OR_PATH, local_files_only=True
            )
            clip_model.vision_model = deepcopy(
                pretrained_clip_vision_models[finetune_mode]
            )
            task_vector = finetuned_clip_vison_models_task_vectors[finetune_mode][
                dataset_name
            ]
            assert set(task_vector.keys()).issubset(
                clip_model.vision_model.state_dict().keys()
            )
            clip_model.vision_model.load_state_dict(
                state_dict_add(
                    clip_model.vision_model.state_dict(),
                    task_vector,
                    strict=False,
                ),
                strict=False,
            )
            # setup fabric modules
            clip_model.vision_model = fabric.setup_module(clip_model.vision_model)
            clip_model.visual_projection = fabric.setup_module(
                clip_model.visual_projection
            )

            datamodule = datamodules[dataset_name]
            test_loader = fabric.setup_dataloaders(test_loaders[dataset_name])
            text = [f"a photo of a {c}" for c in datamodule.classes]

            acc = evaluate_accuracy(
                clip_model=clip_model,
                clip_processor=clip_processor,
                text=text,
                test_loader=test_loader,
            )

            results["task"].append(dataset_name)
            results["finetune_mode"].append(finetune_mode)
            results["step"].append(STEPS)
            results["accuracy"].append(acc)

            print(pd.DataFrame(results))

    results = pd.DataFrame(results)
    results.to_csv("results/ViT-B-16/single_task.csv", index=False)
