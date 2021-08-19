# imports
from sklearn.metrics import classification_report
from data_loaders.table_data_loaders import get_dataloaders
from data_loaders.models import NeuralNet
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import matplotlib
from tableclass_engine import train, validate, \
    f1_nozeros, plot_SA_graph, plot_val_curve
import numpy as np
import json
import os


lrs = [0.001, 0.01, 0.1]
batch_size = [16, 32, 64, 128]
embeds_size = [10, 50, 100]
drop_out = np.arange(0.1, 0.55, 0.05)
hidden_size = [250, 100, 50]
max_len = [100, 300, 500, 1000]

# noinspection PyUnresolvedReferences
matplotlib.style.use('ggplot')

'''
STEPS
# 1. DataLoader
# 2. Multilayer NN. ac func
# 3. loss and optimizer
# 4. training loop,
# 5. model eval,
# 6. GPU support
'''

# ============ Set Seed value =============== #
torch.manual_seed(1)

# ============ Open Config File =============== #
with open("../config/config_tableclass_FFNN.json") as config:
    cf = json.load(config)

# ============ Load and Check Tokenizer =========== #
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=cf["tokenizer_file"])
#tokenizer.add_tokens(["[CAPTION]", "[FIRST_ROW]", "[FIRST_COL]", "[TABLE_BODY]"], special_tokens=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# get vocab size and padding index
vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens) #+ len(tokenizer.get_added_vocab())
padding_idx = tokenizer.pad_token_id

# ============ Get data loaders and datasets =============== #

train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset = get_dataloaders(
    inp_data_dir="../data/train-test-val",
    inp_tokenizer="../tokenizers/tokenizerPKtablesSpecialTokens5000.json",
    max_len=cf["max_len"], batch_size=cf["batch_size"], val_batch_size=cf["val_batch_size"],
    n_workers=cf["n_workers"], remove_html=cf["remove_html"], baseline_only=cf["baseline_only"],
    aug_all=cf["aug_all"], aug_nums=cf["aug_nums"], aug_syns=cf["aug_syns"], aug_both= cf["aug_both"], sampler=cf["sampler"],
    sections=cf["sections_only"],  multi_hot=cf["multi_hot"])

device = 'cpu'


# ============ Define Loss and Optimiser =============== #
all_stats_val = []
all_stats_train = []
all_class_reports = []

for lr in lrs:
    model = NeuralNet(num_classes=cf["num_classes"], embeds_size=cf["embeds_size"],
                      vocab_size=vocab_size, padding_idx=padding_idx, hidden_size=cf["hidden_size"],
                      drop_out=cf["drop_out"]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ============ Train and Val Loop  =============== #
    epochs = cf["epochs"]
    all_train_loss = []
    all_val_loss = []
    all_train_f1_weighted = []
    all_val_f1_weighted = []
    all_train_f1_macro = []
    all_val_f1_macro = []
    best_weightedf1= 0

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

        # update metrics super list
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_f1_weighted.append(train_f1_positives_weighted)
        all_val_f1_weighted.append(val_f1_positives_weighted)
        all_train_f1_macro.append(train_f1_positives_macro)
        all_val_f1_macro.append(val_f1_positives_macro)

    all_stats_train.append(all_train_f1_weighted)
    all_stats_val.append(all_val_f1_weighted)
    all_class_reports.append(val_class_report)


# ========== Plot Results ============#
plot_SA_graph(lrs, all_stats)

max_stats_train= [max(lst) for lst in all_stats_train]
max_stats_val = [max(lst) for lst in all_stats_val]
plot_val_curve(lrs, max_stats)

a=1

# save to file
with open("../data/outputs/val_curve/val_curve.json") as feedsjson:
    feeds = json.load(feedsjson)
    name = str(cf["run_name"], )
    entry = {"name": name}
    feeds.append(entry)
with open("../data/outputs/val_curve/val_curve.json", mode='w') as f:
    f.write(json.dumps(feeds, indent=2))


a=1
