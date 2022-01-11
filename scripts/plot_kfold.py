import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
from typing import Dict, List

matplotlib.style.use('seaborn-whitegrid')


# with open("../data/outputs/val_curve/val_curve.json", mode='w', encoding='utf-8') as f:
# json.dump([], f)

def plot_val_curve(hyperparameters, hyperparm_name, train_scores, train_scores_std, val_scores, val_scores_std):
    plt.figure(figsize=(10, 7))
    plt.plot(hyperparameters, train_scores, label="Training Score", color="darkorange", lw=2)
    plt.fill_between(hyperparameters, train_scores - train_scores_std,
                     train_scores + train_scores_std, alpha=0.2,
                     color="darkorange", lw=2)
    plt.plot(hyperparameters, val_scores, label="Cross Validation Score", color="navy", lw=2)
    plt.fill_between(hyperparameters, val_scores - val_scores_std,
                     val_scores + val_scores_std, alpha=0.2,
                     color="navy", lw=2)
    plt.legend(loc="upper right")
    plt.xlabel(hyperparm_name)
    plt.ylabel('Weighted F1 Score')
    plt.tight_layout()
    plt.show()


with open("../data/outputs/val_curve/val_curve.json") as f:
    vc = json.load(f)

names = [x["run_name"] for x in vc]
print(names)

# "CNN-augnums-resample-kfold-numfilters", "CNN-augnums-resample-kfold-dropouts","CNN-augnums-resample-kfold-embeds",
# "CNN-augnums-resample-kfold-epochs", "CNN-augnums-resample-kfold-batchsize", "CNN-augnums-resample-kfold"
current_run = [x for x in vc if x["run_name"] == "CNN-augnums-resample-kfold-dropouts"]
stats = current_run[0]["summary"]
val = [x["Fw_val"] for x in stats]
# val= val[1:]
val_sd = [x["Fw_val_STD"] for x in stats]
# val_sd= val_sd[1:]
train = [x["Fw_train"] for x in stats]
# train= train[1:]
train_sd = [x["Fw_train_STD"] for x in stats]
# train_sd= train_sd[1:]
lrs = [x["hyperp"] for x in stats]
# lrs= lrs[1:]

val_plot_values = np.array(val)
train_plot_value = np.array(train)
val_plot_sds = np.array(val_sd)
train_plot_sds = np.array(train_sd)
plot_val_curve(lrs, "Dropout", train_plot_value, train_plot_sds, val_plot_values, val_plot_sds)

a = 1
