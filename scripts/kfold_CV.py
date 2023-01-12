# imports
from sklearn.metrics import classification_report
from data_loaders.models import NeuralNet, CNN
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from tableclass_engine import train, validate, \
    f1_nozeros, plot_val_curve
import numpy as np
import pandas as pd
import json
import os
from random import randrange
from data_loaders.bootstrap_data_loaders import read_dataset
from data_loaders.kfold_CV_dataloaders import get_dataloaders
from tqdm import tqdm
import matplotlib

matplotlib.style.use('seaborn-whitegrid')

# ============ Hyperparameters to test =============== #

lrs = [0.05, 0.01, 0.005, 0.001, 0.0005]
num_epochs = [30, 50]
batch_sizes = [16, 32, 64, 128, 192]
embeds_sizes = [10, 50, 100, 200]
drop_outs = [0.0, 0.1, 0.3, 0.5, 0.8]
num_filters = [5, 10, 50, 100, 150, 200]
strides = [1, 2, 3, 4, 5]

# set this each time
hyperparameters = num_epochs
hyperparam_name = "Number of Epochs"

# ============ Set Seed value and Device =============== #
torch.manual_seed(1)
device = 'cpu'

# ============ Open Config File =============== #
with open("../config/config_tableclass_CNN.json") as config:
    cf = json.load(config)

# ============ Load and Check Tokenizer =========== #
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=cf["tokenizer_file"])
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# get vocab size and padding index
vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
padding_idx = tokenizer.pad_token_id

# ============ read in data =============== #
train_samples = list(read_dataset(data_dir_inp="../data/train-test-val", dataset_name="train-corrected"))
valid_samples = list(read_dataset(data_dir_inp="../data/train-test-val", dataset_name="val-corrected"))
train_samples.extend(valid_samples)
train_samples = [{"html": dic["html"], "accept": dic["accept"]} for dic in train_samples]
values = pd.DataFrame(train_samples).values

# ============ Split a dataset into k folds =============== #
k = 5


def cross_validation_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


k_folds = cross_validation_split(train_samples, k)

# ============ Hyperparameter Loop =============== #

val_plot_values = []
val_plot_sds = []
train_plot_value = []
train_plot_sds = []
hyperp_summary = []
for hyperp in hyperparameters:
    # ============ Cross Val Loop =============== #
    k_stats_val = []
    k_stats_train = []
    k_class_stats = []
    k_macro_val = []
    k_micro_val = []
    k_macro_train = []
    k_micro_train = []

    for fold in tqdm(range(len(k_folds))):
        print("\n ===============================")
        print(f"=====K FOLD ITERATION: {fold} =====")
        print("===============================")

        test_list = k_folds[fold]
        train_list = [x for i, x in enumerate(k_folds) if i != fold]
        train_list = [j for i in train_list for j in i]

        train_dataloader, train_dataset, valid_dataloader, valid_dataset = get_dataloaders(
            train_dataset=train_list, val_dataset=test_list,
            inp_tokenizer="../tokenizers/tokenizerPKtablesSpecialTokens5000.json",
            max_len=cf["max_len"], batch_size=cf["batch_size"], val_batch_size=cf["val_batch_size"],
            n_workers=cf["n_workers"], remove_html=cf["remove_html"], baseline_only=cf["baseline_only"],
            aug_all=cf["aug_all"], aug_nums=cf["aug_nums"], aug_syns=cf["aug_syns"], aug_both=cf["aug_both"],
            sampler=cf["sampler"],
            sections=cf["sections_only"], multi_hot=cf["multi_hot"])

        # ============ Get Model =============== #
        # model = NeuralNet(num_classes=cf["num_classes"], embeds_size=cf["embeds_size"],
        # vocab_size=vocab_size, padding_idx=padding_idx, hidden_size=cf["hidden_size"],
        # drop_out=cf["drop_out"]).to(device)

        model = CNN(num_filters=cf["num_filters"], input_channels=cf["input_channels"],
                    num_classes=cf["num_classes"], embeds_size=cf["embeds_size"], filter_sizes=cf["filter_sizes"],
                    vocab_size=vocab_size, padding_idx=padding_idx, stride=cf["stride"], drop_out=cf["drop_out"]).to(
            device)

        # ============ Define Loss and Optimiser =============== #
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cf["lr"])

        # ============ Train and Val Loop  =============== #
        epochs = hyperp
        all_train_loss = []
        all_val_loss = []
        all_train_f1_weighted = []
        all_val_f1_weighted = []
        all_train_f1_macro = []
        all_val_f1_macro = []
        all_train_f1_micro = []
        all_val_f1_micro = []
        best_weightedf1 = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} of {epochs}")

            train_labels, train_predictions, train_loss = train(
                model, train_dataloader, optimizer, criterion, train_dataset, device
            )

            val_labels, val_predictions, val_loss = validate(
                model, valid_dataloader, criterion, valid_dataset, device
            )

            # calculate metrics
            train_class_report = classification_report(train_labels, train_predictions, output_dict=True)
            val_class_report = classification_report(val_labels, val_predictions, output_dict=True)
            train_f1_positives_macro, train_f1_positives_weighted = f1_nozeros(train_class_report)
            val_f1_positives_macro, val_f1_positives_weighted = f1_nozeros(val_class_report)
            train_macro = train_class_report["macro avg"]["f1-score"]
            train_micro = train_class_report["micro avg"]["f1-score"]
            val_macro = val_class_report["macro avg"]["f1-score"]
            val_micro = val_class_report["micro avg"]["f1-score"]

            # update metrics super list
            all_train_f1_weighted.append(train_f1_positives_weighted)
            all_val_f1_weighted.append(val_f1_positives_weighted)
            all_train_f1_macro.append(train_macro)
            all_val_f1_macro.append(val_macro)
            all_train_f1_micro.append(train_micro)
            all_val_f1_micro.append(val_micro)

        k_stats_val.append(all_val_f1_weighted)
        k_stats_train.append(all_train_f1_weighted)
        k_macro_val.append(all_val_f1_macro)
        k_macro_train.append(all_train_f1_macro)
        k_micro_val.append(all_val_f1_micro)
        k_micro_train.append(all_train_f1_micro)

    # average the scores from across the k-folds
    mean_stats_val = np.mean([max(lst) for lst in k_stats_val])
    sd_stats_val = np.std([max(lst) for lst in k_stats_val])
    mean_stats_train = np.mean([max(lst) for lst in k_stats_train])
    sd_stats_train = np.std([max(lst) for lst in k_stats_train])

    mean_macro_val = np.mean([max(lst) for lst in k_macro_val])
    sd_macro_val = np.std([max(lst) for lst in k_macro_val])
    mean_macro_train = np.mean([max(lst) for lst in k_macro_train])
    sd_macro_train = np.std([max(lst) for lst in k_macro_train])

    mean_micro_val = np.mean([max(lst) for lst in k_micro_val])
    sd_micro_val = np.std([max(lst) for lst in k_micro_val])
    mean_micro_train = np.mean([max(lst) for lst in k_micro_train])
    sd_micro_train = np.std([max(lst) for lst in k_micro_train])

    # append to hyperp lists
    val_plot_values.append(mean_stats_val)
    val_plot_sds.append(sd_stats_val)
    train_plot_value.append(mean_stats_train)
    train_plot_sds.append(sd_stats_train)
    hyperp_summary.append({"hyperp": hyperp, "Fw_val": mean_stats_val, "Fw_val_STD": sd_stats_val,
                           "Fw_train": mean_stats_train, "Fw_train_STD": sd_stats_train,
                           "Fma_val": mean_macro_val, "Fma_val_STD": sd_macro_val,
                           "Fma_train": mean_macro_train, "Fma_train_STD": sd_macro_train,
                           "Fmi_val": mean_micro_val, "Fmi_val_STD": sd_micro_val,
                           "Fmi_train": mean_micro_train, "Fmi_train_STD": sd_micro_train})

# ========== Plot Results ============#
val_plot_values = np.array(val_plot_values)
train_plot_value = np.array(train_plot_value)
val_plot_sds = np.array(val_plot_sds)
train_plot_sds = np.array(train_plot_sds)
plot_val_curve(hyperparameters, hyperparam_name, train_plot_value, train_plot_sds, val_plot_values, val_plot_sds)

# ========== Save Results ============#
with open("../data/outputs/val_curve/val_curve.json") as feedsjson:
    feeds = json.load(feedsjson)
    name = str(cf["run_name"])
    entry = {"run_name": name, "summary": hyperp_summary}
    feeds.append(entry)
with open("../data/outputs/val_curve/val_curve.json", mode='w') as f:
    f.write(json.dumps(feeds, indent=2))


