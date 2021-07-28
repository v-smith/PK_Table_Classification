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
from torch.utils.tensorboard import SummaryWriter
from data_loaders.models import CNN

writer = SummaryWriter("../data/runs/")

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
    n_workers=cf["n_workers"], remove_html=cf["remove_html"], baseline_only=cf["baseline_only"], aug_all=False, aug_nums=True, aug_syns=False,
    sampler=True)

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
criterion = nn.BCELoss()  # reduction? weight=weights
#criterion = nn.BCEWithLogitsLoss(pos_weight=weights, reduce=None)
optimizer = torch.optim.Adam(model.parameters(), lr=cf["lr"])

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

    # save loss at each training step
    writer.add_scalar("Loss/train", train_loss, epoch+1)
    writer.add_scalar("Loss/val", val_loss, epoch+1)
    writer.add_scalar("F1_weighted/train", train_f1_positives_weighted, epoch+1)
    writer.add_scalar("F1_weighted/val", val_f1_positives_weighted, epoch+1)

    #save checkpoint
    is_best = val_f1_positives_weighted > best_weightedf1
    best_weightedf1 = max(val_f1_positives_weighted, best_weightedf1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_f1': best_weightedf1,
        'optimizer': optimizer.state_dict(),
    }, is_best, cf["run_name"])

    # update metrics super list
    all_train_loss.append(train_loss)
    all_val_loss.append(val_loss)
    all_train_f1_weighted.append(train_f1_positives_weighted)
    all_val_f1_weighted.append(val_f1_positives_weighted)
    all_train_f1_macro.append(train_f1_positives_macro)
    all_val_f1_macro.append(val_f1_positives_macro)

#make sure that all pending events have been written to disk
writer.flush()
#close writer
#writer.close()

# ========== Plot Results and Save/ Tensor Board ============#

class_report = classification_report(val_labels, val_predictions)
print("==== FINAL VALIDATION CLASS REPORT ====")
print(class_report)
plot_loss_graph(all_train_loss, all_val_loss, cf)
plot_f1_graph(all_train_f1_macro, all_val_f1_macro, cf, "- Positive Classes")
plot_f1_graph(all_train_f1_weighted, all_val_f1_weighted, cf, "- Weighted Positive Classes")

a=1
