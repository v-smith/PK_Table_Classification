# imports
from sklearn.metrics import classification_report
from data_loaders.table_data_loaders import get_dataloaders
from data_loaders.table_data_loaders import NeuralNet
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

# ============ Define Loss and Optimiser =============== #
criterion = nn.BCELoss()  # functional.binary_cross_entropy
optimizer = torch.optim.Adam(model.parameters(), lr=cf["lr"])

# ============ Train and Val Loop  =============== #

epochs = cf["epochs"]
all_val_loss= []
all_val_f1= []
all_f1_macro = []
all_f1_weighted = []
all_f1_macro_positives = []
all_f1_weighted_positives = []
all_f1_notrel = []
all_loss= []

first_batch = next(iter(test_dataloader))
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")

    labels, predictions, loss = overfit_subset(
        model, first_batch, optimizer, criterion, device)

    # calculate metrics
    overfit_class_report = classification_report(labels, predictions, output_dict=True)
    f1_macro, f1_weighted = f1_nozeros(overfit_class_report, remove_nonrel=False, only_notrel=False)
    f1_macro_positives, f1_weighted_positives = f1_nozeros(overfit_class_report, remove_nonrel=True, only_notrel=False)
    f1_macro_notrel, f1_weighted_notrel = f1_nozeros(overfit_class_report, remove_nonrel=False, only_notrel=True)

    # update metrics super list
    all_f1_macro.append(f1_macro)
    all_f1_weighted.append(f1_weighted)
    all_f1_macro_positives.append(f1_macro_positives)
    all_f1_weighted_positives.append(f1_weighted_positives)
    all_f1_notrel.append(f1_macro_notrel)
    all_loss.append(loss)

# plot and save the train and validation line graphs
plot_loss_graph(all_loss, all_val_loss, cf)
plot_f1_graph(all_f1_macro, all_val_f1, cf, "All Macro")
plot_f1_graph(all_f1_weighted, all_val_f1, cf, "All Weighted")
plot_f1_graph(all_f1_macro_positives, all_val_f1, cf, "All Positive Macro")
plot_f1_graph(all_f1_weighted_positives, all_val_f1, cf, "All Positive Weighted")
plot_f1_graph(all_f1_notrel, all_val_f1, cf, "Not Relevant Only")

# ========== Tensor Board ============#
# TODO: Implement Tensorboard
# tensorboard
# writer.add_scalar('training loss', train_running_loss / counter, n_total_steps + i)

# img_grid = torchvision.utils.make_grid()
# writer.add_image("tableclass", img_grid)
# writer.close()

# ============ Save the Best Model  =============== #

# TODO: save best model- look at: https://github.com/lavis-nlp/spert/blob/master/spert/trainer.py
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, ('../data/outputs/model_saves/' + cf["run_name"] + ".pth"))

# ============ Test Loop  =============== #

# load the model checkpoint
model_checkpoint = torch.load('../data/outputs/model_saves/trial-FFNN-classification.pth')

# load model weights state_dict and optimizer state dict
model.load_state_dict(model_checkpoint['model_state_dict'])
optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])

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
global_metrics = label_wise_metrics(all_labels, all_predictions)
class_report = classification_report(labels, preds)
class_report_dict = classification_report(labels, preds, output_dict=True)
f1_nozeros = f1_nozeros(class_report_dict)
f1 = f1_all(class_report_dict)

# prints
print(f"Global Metrics: {global_metrics}")
print(f"All F1: {f1} | "
      f"No Zeros F1: {f1_nozeros}")
print(f"Classification Report: {class_report}")
a = 1
