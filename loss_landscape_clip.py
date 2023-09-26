# %%
import loss_landscapes
import numpy as np
import torch.nn.functional as F
from torch import nn

from evaluate_single_task_clip import *

# %%
fabric = L.Fabric(accelerator="gpu", devices=1)
fabric.launch()


# %%
def evaluate_loss(
    *,
    clip_model: CLIPModel,
    clip_processor,
    text: List[str],
    test_loader,
) -> float:
    clip_model.eval()
    # precompute the text features
    text_input = clip_processor(text, return_tensors="pt", padding=True)
    text_embeds = clip_model.get_text_features(**text_input)

    loss, count = 0, 0

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

                loss += F.cross_entropy(
                    logits_per_image, labels, reduction="sum"
                ).item()
                count += len(labels)

    return loss / count


def evaluate_loss_for_clip_vision_model(
    clip_vision_model, test_loader: DataLoader, classes: List[str]
) -> float:
    # replace the `clip_model.vision_model` with `clip_vision_model`
    clip_processor, clip_model = load_clip_model(
        MODEL_NAME_OR_PATH, local_files_only=True
    )
    clip_model.vision_model = deepcopy(clip_vision_model)

    # setup fabric modules
    clip_model.vision_model = fabric.setup_module(clip_model.vision_model)
    clip_model.visual_projection = fabric.setup_module(clip_model.visual_projection)

    # compute text features
    text = [f"a photo of a {c}" for c in classes]
    test_loader = fabric.setup_dataloaders(test_loader)

    loss = evaluate_loss(
        clip_model=clip_model,
        clip_processor=clip_processor,
        text=text,
        test_loader=test_loader,
    )

    return loss


# %%
from loss_landscapes.metrics import Metric


class SingleTaskLoss(Metric):
    def __init__(self, task_name: str):
        super().__init__()
        self.task_name = task_name
        self.dataloader = test_loaders[task_name]
        self.classes = datamodules[task_name].classes

    def __call__(self, model_wrapper: loss_landscapes.ModelWrapper) -> float:
        # print(model_wrapper.modules)
        loss = evaluate_loss_for_clip_vision_model(
            model_wrapper.modules[0],
            self.dataloader,
            self.classes,
        )
        return loss


class MultiTaskLoss(Metric):
    def __init__(self, task_names: List[str]):
        super().__init__()
        self.task_names = task_names
        self.dataloaders = [test_loaders[t] for t in task_names]
        self.classes = [datamodules[t].classes for t in task_names]

    def __call__(self, model_wrapper: loss_landscapes.ModelWrapper) -> float:
        loss = 0
        for dataloader, classes in zip(self.dataloaders, self.classes):
            loss += evaluate_loss_for_clip_vision_model(
                model_wrapper.modules[0],
                dataloader,
                classes,
            )
        return loss


def dummy_loss(*args):
    return 1


# %%
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(usage=str(DATASET_NAMES))
    parser.add_argument(
        "task_one",
        type=str,
    )
    parser.add_argument(
        "task_two",
        type=str,
    )
    args = parser.parse_args()

    task_one = args.task_one
    task_two = args.task_two

    #     |
    #     |
    # --- 0 --- 2
    #     |
    #     1
    model_start: nn.Module = deepcopy(pretrained_clip_vision_models["standard"])
    model_end_one: nn.Module = deepcopy(pretrained_clip_vision_models["standard"])
    model_end_two: nn.Module = deepcopy(pretrained_clip_vision_models["standard"])
    model_start.load_state_dict(
        state_dict_add(
            model_start.state_dict(),
            state_dict_add(
                finetuned_clip_vison_models_task_vectors["standard"][task_one],
                finetuned_clip_vison_models_task_vectors["standard"][task_two],
            ),
            strict=False,
        )
    )
    model_end_one.load_state_dict(
        state_dict_add(
            model_end_one.state_dict(),
            state_dict_mul(
                finetuned_clip_vison_models_task_vectors["standard"][task_one], 2.0
            ),
            strict=False,
        )
    )
    model_end_two.load_state_dict(
        state_dict_add(
            model_end_two.state_dict(),
            state_dict_mul(
                finetuned_clip_vison_models_task_vectors["standard"][task_two], 2.0
            ),
            strict=False,
        )
    )

    landscape = loss_landscapes.planar_interpolation(
        model_start=model_start,
        model_end_one=model_end_one,
        model_end_two=model_end_two,
        metric=MultiTaskLoss([task_one, task_two]),
        deepcopy_model=False,
    )
    np.save(f"results/ViT-B-16/landscape_{task_one}-{task_two}.npy", landscape)
