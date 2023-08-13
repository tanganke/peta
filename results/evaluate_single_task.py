"""
load data from `peta.xlsx`, sheet `finetuned performance`

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

# %%
# load data
data = pd.read_excel("peta.xlsx", sheet_name="finetuned performance")

# %%
# process data
for row_id in range(len(data)):
    config = DictConfig(eval(data.at[row_id, "config"]))
    if config.peft.peft_config is not None:
        data.at[row_id, "RoLA.r"] = config.peft.peft_config.r
    else:
        data.at[row_id, "RoLA.r"] = math.nan
    data.at[row_id, "batch_size"] = config.batch_size
    data.at[row_id, "steps"] = config.trainer.max_steps
    data.at[row_id, "lr"] = config.optim.optimizer.lr
    data.at[row_id, "weight_decay"] = config.optim.optimizer.weight_decay

# %%
data = data.sort_values(by=["Model", "Task", "Method", "config"])
data.to_csv("single_task.csv", index=False)

# %%
"""    
plot the results as bar chart, the y axis is `Accuracy`, the x axis is `Task` and `Method`
cheat the same Task together, assign different color to different Method.
Each method runs under different hyperparameters, so the same method may have different accuracy, choose the maximum accuracy.
"""
max_data = data.groupby(["Task", "Method"]).max("Accuracy").reset_index()

plt.figure(figsize=(14, 5))
ax = sns.barplot(data=max_data, x="Task", y="Accuracy", hue="Method")
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
    data=data, x="Task", y="Accuracy", hue="Method", errorbar="sd"
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
