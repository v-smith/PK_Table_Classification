import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
from typing import Dict, List

matplotlib.style.use('seaborn-whitegrid')

with open("../data/outputs/bootstrap/bootstrap.json") as f:
    bs = json.load(f)

names= [x["name"] for x in bs]
print(names)


def plot_bootstrap_hist(class_stats: Dict):
    #data = class_stats["f1s"]
    data=class_stats["bootstrap_results"]
    #num = class_stats["class"]
    label_list = ["NC Params", "C Params", "Par_Cov Rs", "Params Other",
                  "Doses", "# of subjects", "Sample timings", "Demographics", "Covs other"]
    #label = label_list[num]
    n = list(np.arange(min(data), (max(data) + 0.01), 0.01))
    plt.hist(data, bins=n)
    #plt.title(f"Bootstrap Distribution for Class {label}")
    # plt.savefig("../data/outputs/bootstrap/hists/" + str(cf["run_name"]))
    plt.show()

def box_plot_all(metrics_df, columns, scoring):
    plt.boxplot(metrics_df, labels=columns, showfliers=False)
    plt.xlabel("Architecture")
    plt.xticks(rotation=90)
    plt.ylabel(scoring)
    plt.tight_layout()
    plt.show()


current_run = [x for x in bs if x["name"] == 'FFNN-BOW-100']
per_class_stats = current_run[0]["per_class_stats_ready"]

#ordered list of runs wanted
desired_columns= ['FFNN-boostrap-macros100', 'FFNN-boostrap-resample-macros100',
                  'FFNN-boostrap-augsyns-macros100', 'FFNN-boostrap-augnums-macros100',
                  'FFNN-boostrap-augall-macros100',
                  'CNN-macro-bootstrap100','CNN-resample-macro-bootstrap100',
                  'CNN-augsyns-macro-bootstrap100','CNN-augnums-macro-bootstrap100',
                  'CNN-AugNums-Resample-bootstrap100', 'CNN-augboth-macro-bootstrap100',
                  'FFNN_Multihot-boostrap-20','CNN-Mutlihot-bootstrap20']

desired_bow= ['FFNN-BOW-100', 'FFNN-boostrap-macros100']

runs= [x for x in bs if x["name"] in desired_columns]
weightedf1s= [{x["name"]: x["bootstrap_results"]} for x in runs]
w_merged = {}
for d in weightedf1s:
    w_merged.update(d)
replace_ffnn= 5 * w_merged["FFNN_Multihot-boostrap-20"]
replace_cnn= 5 * w_merged['CNN-Mutlihot-bootstrap20']
w_merged["FFNN_Multihot-boostrap-20"]= replace_ffnn
w_merged['CNN-Mutlihot-bootstrap20']=replace_cnn
macrof1s= [{x["name"]:x["bootstrap_macro"]} for x in runs]
macro_merged = {}
for d in macrof1s:
    macro_merged.update(d)
replace_ffnn_m= 5 * macro_merged["FFNN_Multihot-boostrap-20"]
replace_cnn_m= 5 * macro_merged['CNN-Mutlihot-bootstrap20']
macro_merged["FFNN_Multihot-boostrap-20"]= replace_ffnn_m
macro_merged['CNN-Mutlihot-bootstrap20']=replace_cnn_m

weighted_df= pd.DataFrame(w_merged)
weighted_df.columns= ["MLP-Mutlihot", "CNN-Mutlihot", 'CNN-Nums-Undersample', 'MLP',
       'MLP-Undersample', 'MLP-Nums',
       'MLP-Syns', 'MLP-Nums&Syns',
       'CNN','CNN-Undersample',
       'CNN-Nums','CNN-Syns',
       'CNN-Nums&Syns']
weighted_df= weighted_df.reindex(sorted(weighted_df.columns), axis=1)
macros_df= pd.DataFrame(macro_merged)
macros_df.columns= ["MLP-Mutlihot", "CNN-Mutlihot", 'CNN-Nums-Undersample', 'MLP',
       'MLP-Undersample', 'MLP-Nums',
       'MLP-Syns', 'MLP-Nums&Syns',
       'CNN','CNN-Undersample',
       'CNN-Nums','CNN-Syns',
       'CNN-Nums&Syns']
macros_df = macros_df.reindex(sorted(macros_df.columns), axis=1)
box_plot_all(metrics_df=weighted_df, columns=weighted_df.columns, scoring="Weighted F1")
box_plot_all(metrics_df=macros_df, columns=macros_df.columns, scoring="Macro F1")
a=1

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

#for x in per_class_stats:
    #plot_bootstrap_hist(x)


median= current_run[0]["median"]
print(f"median: {median} ")
upper= current_run[0]["upper_CI"]
print(f"upper: {upper} ")
lower= current_run[0]["lower_CI"]
print(f"lower: {lower}")

median_m= current_run[0]["median_macro"]
print(f"mac_median: {median_m} ")
upper_m= current_run[0]["upper_percent_macro"]
print(f"mac_upper: {upper_m} ")
lower_m= current_run[0]["lower_percent_macro"]
print(f"mac_lower: {lower_m}")

median_mi= current_run[0]["median_micro"]
print(f"median_micro: {median_mi} ")
upper_mi= current_run[0]["upper_percent_micro"]
print(f"upper: {upper_mi} ")
lower_mi= current_run[0]["lower_percent_micro"]
print(f"lower: {lower_mi}")

print_stats(per_class_stats)
a = 1
