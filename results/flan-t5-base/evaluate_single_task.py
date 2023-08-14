"""
load data from `(fullfinetuned)|(lora)|(l_lora)_results_v{version}.csv`, gather the results of single task evaluation， save to `signle_task.csv'.

the `config` column is a DictConfig string, which contains the config of the model
extract some hyperparameters from `config` column and add them to the dataframe.
"""
# %%
import functools
import json
import math
import os
import sys
import seaborn as sns
import itertools

from IPython.display import display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# %%
# load data
all_data = None
for method in ["fullfinetuned", "lora", "l_lora"]:
    for version in itertools.count():
        csv_file = f"{method}_results_v{version}.csv"
        if not os.path.exists(csv_file):
            break
        else:
            data = pd.read_csv(csv_file)
            # set coulmn `method` to `method`
            data["method"] = method
            data["version"] = version
            data = data.drop_duplicates(ignore_index=True)
            print(f"load {csv_file}")
            if all_data is None:
                all_data = data
            else:
                all_data = pd.concat([all_data, data], ignore_index=True)

display(all_data.head())
# %%
# replace glue-stsb accuracy with spearman rho, data is loaded from `{method}_results_glue-stsb_v{version}.csv`
for row_id in range(len(all_data)):
    if all_data.at[row_id, "dataset"] == "glue-stsb":
        method = all_data.at[row_id, "method"]
        version = all_data.at[row_id, "version"]
        data = pd.read_csv(f"{method}_results_glue-stsb_v{version}.csv")
        assert len(data) == 1
        all_data.at[row_id, "accuracy"] = data.at[0, "accuracy"]


# %%
data = all_data
# drop column `Unnamed: 0`
data = data.drop(columns=["Unnamed: 0"])
# drop duplicate rows
data = data.drop_duplicates()
# %%
# process data
for row_id in range(len(data)):
    config = DictConfig(eval(data.at[row_id, "config"]))
    if config.peft.peft_config is not None:
        data.at[row_id, "LoRA.r"] = config.peft.peft_config.r
    else:
        data.at[row_id, "LoRA.r"] = math.nan
    data.at[row_id, "batch_size"] = config.batch_size
    data.at[row_id, "steps"] = config.trainer.max_steps
    data.at[row_id, "lr"] = config.optim.optimizer.lr
    data.at[row_id, "weight_decay"] = config.optim.optimizer.weight_decay

# %%
data = data.sort_values(by=["model", "dataset", "method", "config"])
data.to_csv("single_task.csv", index=False)

# %%
"""    
plot the results as bar chart, the y axis is `Accuracy`, the x axis is `Task` and `Method`
cheat the same Task together, assign different color to different Method.
Each method runs under different hyperparameters, so the same method may have different accuracy, choose the maximum accuracy.
"""
max_data = data.groupby(["dataset", "method"]).max("accuracy").reset_index()

plt.figure(figsize=(14, 5))
ax = sns.barplot(data=max_data, x="dataset", y="accuracy", hue="method")
for p in ax.containers:
    ax.bar_label(
        p,
        labels=["{:.1f}%".format(val.get_height() * 100) for val in p],
        label_type="edge",
        fontsize=8,
    )
plt.title("single task performance (max)")
plt.savefig("single_task_max.pdf")
plt.show()

# standard derivation as errorbar
plt.figure(figsize=(14, 5))
ax: plt.Axes = sns.barplot(
    data=data, x="dataset", y="accuracy", hue="method", errorbar="sd"
)
for p in ax.containers:
    ax.bar_label(
        p,
        labels=["{:.1f}%".format(val.get_height() * 100) for val in p],
        label_type="edge",
        fontsize=8,
    )
plt.title("single task performance")
# save pdf
plt.savefig("single_task.pdf")
plt.show()


# %%
