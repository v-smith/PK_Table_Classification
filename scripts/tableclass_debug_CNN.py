# imports
from sklearn.metrics import classification_report
from data_loaders.table_data_loaders import get_dataloaders
from data_loaders.models import CNN
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import matplotlib
from tableclass_engine import label_wise_metrics, \
    plot_loss_graph, plot_f1_graph, f1_nozeros, overfit_subset
import numpy as np
import json
import os
from torch.utils.tensorboard import SummaryWriter

# noinspection PyUnresolvedReferences
matplotlib.style.use('ggplot')

# ============ Set Seed value =============== #
torch.manual_seed(1)

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

# ============ Get data loaders and datasets =============== #

train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset = get_dataloaders(
    inp_data_dir="../data/train-test-val",
    inp_tokenizer="../tokenizers/tokenizerPKtablesSpecialTokens5000.json",
    max_len=cf["max_len"], batch_size=cf["batch_size"], val_batch_size=cf["val_batch_size"],
    n_workers=cf["n_workers"], baseline_only=False, remove_html=True)

# ============ Set Device =============== #
# device config
device = 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)

# ============ Get Model =============== #
model = CNN(seq_len=cf["seq_len"], out_channels=cf["out_channels"], input_channels=cf["input_channels"],
            num_classes=cf["num_classes"], embeds_size=cf["embeds_size"], kernel_heights=cf["kernel_heights"],
            vocab_size=vocab_size, padding_idx=padding_idx, stride=cf["stride"]).to(device)

# ============ Define Loss and Optimiser =============== #
criterion = nn.BCELoss()  # functional.binary_cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=cf["lr"])

# ============ Train and Val Loop  =============== #

epochs = cf["epochs"]
all_val_loss= []
all_val_f1= []
all_f1_macro_positives = []
all_f1_weighted_positives = []
all_loss= []

first_batch = next(iter(test_dataloader))
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")

    labels, predictions, loss = overfit_subset(
        model, first_batch, optimizer, criterion, device)

    # calculate metrics
    overfit_class_report = classification_report(labels, predictions, output_dict=True)
    f1_macro_positives, f1_weighted_positives = f1_nozeros(overfit_class_report)

    # update metrics super list
    all_f1_macro_positives.append(f1_macro_positives)
    all_f1_weighted_positives.append(f1_weighted_positives)
    all_loss.append(loss)

# plot and save the train and validation line graphs
plot_loss_graph(all_loss, all_val_loss, cf)
plot_f1_graph(all_f1_macro_positives, all_val_f1, cf, "All Positive Macro")
plot_f1_graph(all_f1_weighted_positives, all_val_f1, cf, "All Positive Weighted")

a=1