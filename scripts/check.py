import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import jsonlines
import json
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Using device:', device)

with open("../data/outputs/bootstrap/bootstrap.json") as f:
    bs = json.load(f)

max_stats = [x["bootstrap_results"] for x in bs if x["name"] == "BootstrapTest-"]
data = max_stats[0]
#data = [round(y, 2) for y in data]
n = list(np.arange(min(data), (max(data) + 0.005), 0.005))
plt.hist(data, bins=n)
plt.show()
a=1

