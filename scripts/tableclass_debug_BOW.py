# imports
from sklearn.metrics import classification_report
from data_loaders.bow_data_loaders import get_dataloaders
from data_loaders.models import BOW_NeuralNet
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import matplotlib
from tableclass_engine import plot_loss_graph, plot_f1_graph, f1_nozeros, overfit_subset
import json
import os

# noinspection PyUnresolvedReferences
matplotlib.style.use('ggplot')

# ============ Set Seed value =============== #
torch.manual_seed(1)

# ============ Open Config File =============== #
with open("../config/config_tablesclass_BOW.json") as config:
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
    inp_data_dir="../data/train-test-val", batch_size=cf["batch_size"],
    n_workers=cf["n_workers"], remove_html=cf["remove_html"], baseline_only=cf["baseline_only"],
    aug_all=cf["aug_all"], aug_nums=cf["aug_nums"], aug_syns=cf["aug_syns"])

# ============ Set Device =============== #
# device config
device = 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.autograd.set_detect_anomaly(True)

# ============ Get Model =============== #
model = BOW_NeuralNet(num_classes=cf["num_classes"], input_size=cf["input_size"],
                      hidden_size=cf["hidden_size"], drop_out=cf["drop_out"]).to(device)

# ============ Define Loss and Optimiser =============== #
criterion = nn.BCELoss()  # functional.binary_cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=cf["lr"])

# ============ Train and Val Loop  =============== #

epochs = cf["epochs"]
all_val_loss = []
all_val_f1 = []
all_f1_macro = []
all_f1_weighted = []
all_f1_macro_positives = []
all_f1_weighted_positives = []
all_f1_notrel = []
all_loss = []

first_batch = next(iter(test_dataloader))
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")

    labels, predictions, loss = overfit_subset(
        model, first_batch, optimizer, criterion, device)

    # calculate metrics
    overfit_class_report = classification_report(labels, predictions, output_dict=True)
    f1_macro, f1_weighted = f1_nozeros(overfit_class_report)

    # update metrics super list
    all_f1_macro.append(f1_macro)
    all_f1_weighted.append(f1_weighted)
    all_loss.append(loss)

# plot and save the train and validation line graphs
print("==== FINAL VALIDATION CLASS REPORT ====")
plot_loss_graph(all_loss, all_val_loss, cf)
plot_f1_graph(all_f1_macro, all_val_f1, cf, "- Positive Classes")
plot_f1_graph(all_f1_weighted, all_val_f1, cf, "- Weighted Positive Classes")

a = 1
