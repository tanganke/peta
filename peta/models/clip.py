import os
from typing import Optional

import lightning.pytorch as pl
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor

CLIP_MODELS = [
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
]


def load_clip_model(model_name_or_path: str, local_files_only: bool = True):
    processor = CLIPProcessor.from_pretrained(
        model_name_or_path, local_files_only=local_files_only
    )
    clip_model: CLIPModel = CLIPModel.from_pretrained(
        model_name_or_path, local_files_only=local_files_only
    )
    return processor, clip_model


def freeze_unless_image_model(clip_model: CLIPModel):
    """
    Freezes the parameters in a CLIP model instance other than image model.

    Args:
        clip_model (CLIPModel): A CLIP model instance.

    Returns:
        CLIPModel: The same CLIP model instance with the text model parameters frozen.
    """
    for param in clip_model.parameters():
        param.requires_grad = False
    for param in clip_model.vision_model.parameters():
        param.requires_grad = True
    for param in clip_model.visual_projection.parameters():
        param.requires_grad = True
    return clip_model


def get_lora_vision_model(clip_model: CLIPModel, lora_config: LoraConfig):
    """
    Returns a PEFT model for the vision part of a CLIP model, based on the given LoraConfig.

    >>> peft_config = LoraConfig(
            target_modules=["v_proj", "q_proj"],
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )

    Args:
        clip_model (CLIPModel): A CLIP model instance.
        lora_config (LoraConfig): A LoraConfig instance with the desired PEFT configuration.

    Returns:
        PeftModel: A PEFT model instance for the vision part of the CLIP model.
    """
    vision_model = clip_model.vision_model
    lora_vision_model = get_peft_model(vision_model, lora_config)
    return lora_vision_model


if __name__ == "__main__":
    lora_config = LoraConfig(
        target_modules=["v_proj", "q_proj"],
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    for model_name in CLIP_MODELS:
        preprocess, clip_model = load_clip_model(model_name)
        lora_vison_model = get_lora_vision_model(clip_model, lora_config)
