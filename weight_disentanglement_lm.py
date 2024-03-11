# import state dicts and so on from evaluate_multi_task.py
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

from evaluate_multi_task_lm import *
from peta.tasks.ties_merging import (
    normalize,
    state_dict_to_vector,
    vector_to_state_dict,
)

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 12


def error_rate(y_true, y_pred):
    error = 0
    count = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            error += 1
        count += 1
    return error / count


@torch.no_grad()
def gather_predicts(model: nn.Module, dataloader: DataLoader):
    predictions = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 20:
            break
        outputs = model.generate(batch["input_ids"])
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(output_text)
    return predictions


# ------------------------------------------
# experimental settings
torch.set_float32_matmul_precision("medium")
task_0 = "glue-cola"
task_1 = "glue-rte"
finetune_mode = "l_lora"
# ------------------------------------------

if finetune_mode == "lora":
    pretrained_model = lora_pretrained_model
    task_vector = lora_task_vector
elif finetune_mode == "l_lora":
    pretrained_model = l_lora_pretrained_model
    task_vector = l_lora_task_vector
else:
    raise ValueError(f"unknown finetune_mode: {finetune_mode}")

n = 21
result = np.zeros((n, n), dtype=np.float32)

predicts_0 = {}
for lambda_0 in tqdm(np.linspace(-2, 2, n)):
    model: nn.Module = deepcopy(pretrained_model)
    model.load_state_dict(
        state_dict_add(
            model.state_dict(),
            state_dict_mul(task_vector[task_0], lambda_0),
            strict=False,
        ),
        strict=False,
    )
    model = fabric.setup_module(model)
    model.eval()
    predicts_0[lambda_0] = gather_predicts(model, val_loaders[task_0])

predicts_1 = {}
for lambda_1 in tqdm(np.linspace(-2, 2, n)):
    model: nn.Module = deepcopy(pretrained_model)
    model.load_state_dict(
        state_dict_add(
            model.state_dict(),
            state_dict_mul(task_vector[task_1], lambda_1),
            strict=False,
        ),
        strict=False,
    )
    model = fabric.setup_module(model)
    model.eval()
    predicts_1[lambda_1] = gather_predicts(model, val_loaders[task_1])

for i, lambda_0 in enumerate(tqdm(np.linspace(-2, 2, n))):
    for j, lambda_1 in enumerate(np.linspace(-2, 2, n)):
        model: nn.Module = deepcopy(pretrained_model)
        model.load_state_dict(
            state_dict_add(
                model.state_dict(),
                state_dict_add(
                    state_dict_mul(task_vector[task_0], lambda_0),
                    state_dict_mul(task_vector[task_1], lambda_1),
                ),
                strict=False,
            ),
            strict=False,
        )
        model = fabric.setup_module(model)
        model.eval()

        predicts_on_0 = gather_predicts(model, val_loaders[task_0])
        predicts_on_1 = gather_predicts(model, val_loaders[task_1])

        result[i, j] = error_rate(
            predicts_0[lambda_0],
            predicts_on_0,
        ) + error_rate(
            predicts_1[lambda_1],
            predicts_on_1,
        )

np.save(
    f"results/flan-t5-base/{finetune_mode}_weight_disentanglement-{task_0}-{task_1}.npy",
    result,
)
