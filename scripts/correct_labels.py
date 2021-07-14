# imports
from sklearn.metrics import classification_report
from data_loaders.table_data_loaders import get_dataloaders
from data_loaders.models import NeuralNet
import torch
import jsonlines
from transformers import PreTrainedTokenizerFast
import matplotlib
from tableclass_engine import train, validate, label_wise_metrics, \
    plot_loss_graph, plot_f1_graph, f1_nozeros, save_checkpoint
import numpy as np
import json
import os
from prodigy.components.db import connect
import prodigy
from sklearn.preprocessing import MultiLabelBinarizer


inp_labels = [[1,2,3,4,5,6,7,8,9,10], [4,2], [5,5,5], [2,5]]
new_labels = [list(filter(lambda a: a != 5, x)) for x in inp_labels]
print(new_labels)

mlb = MultiLabelBinarizer()
all_transformed_labels = mlb.fit_transform(new_labels)
all_transformed_labels = list(all_transformed_labels)
print(all_transformed_labels)
a=1
