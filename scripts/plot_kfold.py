import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
from typing import Dict, List
from tableclass_engine import train, validate, f1_nozeros, plot_val_curve


#with open("../data/outputs/val_curve/val_curve.json", mode='w', encoding='utf-8') as f:
    #json.dump([], f)


with open("../data/outputs/val_curve/val_curve.json") as f:
    vc = json.load(f)

names= [x["run_name"] for x in vc]
print(names)

current_run = [x for x in vc if x["run_name"] == "CNN-Lrs-ValCurve-"]
stats = current_run[0]["summary"]
val= [x["Fw_val"] for x in stats]
#val= val[1:]
val_sd= [x["Fw_val_STD"] for x in stats]
#val_sd= val_sd[1:]
train= [x["Fw_train"] for x in stats]
#train= train[1:]
train_sd= [x["Fw_train_STD"] for x in stats]
#train_sd= train_sd[1:]
lrs= [x["lr"] for x in stats]
#lrs= lrs[1:]

val_plot_values= np.array(val)
train_plot_value= np.array(train)
val_plot_sds= np.array(val_sd)
train_plot_sds= np.array(train_sd)
plot_val_curve(lrs, "Learning Rate", train_plot_value, train_plot_sds, val_plot_values, val_plot_sds)

a=1
