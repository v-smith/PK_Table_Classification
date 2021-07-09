# imports
from sklearn.metrics import classification_report
from data_loaders.table_data_loaders import get_dataloaders
from data_loaders.models import NeuralNet
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import matplotlib
from tableclass_engine import train, validate, label_wise_metrics, \
    plot_loss_graph, plot_f1_graph, f1_nozeros, save_checkpoint
import numpy as np
import json
import os

# ============ Open Config File =============== #
with open("../config/config_tableclass_FCNN.json") as config:
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
    n_workers=cf["n_workers"])

# ============ Set Device =============== #
# device config
device = 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.autograd.set_detect_anomaly(True)

# ============ Get Model =============== #

model = NeuralNet(num_classes=cf["num_classes"], embeds_size=cf["embeds_size"],
                  vocab_size=vocab_size, padding_idx=padding_idx, hidden_size=cf["hidden_size"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cf["lr"])

# load the model checkpoint
model_checkpoint = torch.load("../data/outputs/model_saves/trial-FFNN-classificationmodel_best.pth.tar")

# load model weights state_dict and optimizer state dict
model.load_state_dict(model_checkpoint['state_dict'])
optimizer.load_state_dict(model_checkpoint["optimizer"])

# ============ Test Loop  =============== #
# model in eval mode
model.eval()
with torch.no_grad():
    print("==== Begin Evaluation ====")
    all_preds = []
    all_labels = []
    for i, batch in enumerate(test_dataloader):
        input_ids, target = batch['input_ids'].to(device), batch['labels']

        outputs = model(input_ids)
        outputs = torch.sigmoid(outputs).detach().cpu()
        predicted = torch.round(outputs)

        labels = np.asarray(target)
        all_labels.extend(labels)
        preds = np.asarray(predicted)
        all_preds.extend(preds)

# metric calculations
all_labels = np.stack(all_labels, axis=0)
all_predictions = np.stack(all_preds, axis=0)

test_class_dict = classification_report(all_labels,all_preds, output_dict=True)
test_class_report = classification_report(all_labels,all_preds)
test_f1_positives_macro, test_f1_positives_weighted = f1_nozeros(test_class_dict, remove_nonrel=True, only_notrel=False)

# prints
print(f"F1 Weighted (Positive Classes Only): {test_f1_positives_weighted} | "
      f"F1 not weighted (Macro and Positive Classes Only): {test_f1_positives_macro}")
print(f"Classification Report: {test_class_report}")
a = 1
