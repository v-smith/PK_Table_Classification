import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
from typing import Dict, List

with open("../data/outputs/bootstrap/bootstrap.json") as f:
    bs = json.load(f)


def plot_bootstrap_hist(class_stats: Dict):
    data = class_stats["f1s"]
    num = class_stats["class"]
    label_list = ["NC Params", "C Params", "Par_Cov Rs", "Params Other",
                  "Doses", "# of subjects", "Sample timings", "Demographics", "Covs other"]
    label = label_list[num]
    n = list(np.arange(min(data), (max(data) + 0.01), 0.01))
    plt.hist(data, bins=n)
    plt.title(f"Bootstrap Distribution for Class {label}")
    # plt.savefig("../data/outputs/bootstrap/hists/" + str(cf["run_name"]))
    plt.show()


current_run = [x for x in bs if x["name"] == "CNN-NGRAM-Bootstrap-AugAll-100"]
per_class_stats = current_run[0]["per_class_stats_ready"]

under_100s = []
over_100s = []
for x in per_class_stats:
    if np.median(x["supports"]) <= 100:
        under_100s.append(x)
    else:
        over_100s.append(x)


def box_plot(under_100s: List[Dict], over_100s: List[Dict]):
    a = [x["f1s"] for x in under_100s]
    a = [item for sublist in a for item in sublist]
    c = [x["f1s"] for x in over_100s]
    c = [item for sublist in c for item in sublist]
    df = pd.DataFrame(list(zip(a, c)), columns=["Under 100", "200-500"])
    # abc = np.array((a, c)).T
    plt.boxplot(df, labels=["Under 100", "200-500"])
    plt.xlabel("Number of Samples per Class")
    plt.ylabel("F1 Score")
    plt.title("Sensitivity Analysis for number of samples per class versus F1-Score")
    plt.show()


def print_stats(per_class_stats):
    for x in per_class_stats:
        print(x["class"],
              round(x["lower_CI"] *100, 1),
              round(x["upper_CI"] *100, 1),
              round(x["median"]*100, 1),
              np.median(x["supports"]))

for x in per_class_stats:
    plot_bootstrap_hist(x)

print_stats(per_class_stats)
box_plot(under_100s, over_100s)
a = 1
