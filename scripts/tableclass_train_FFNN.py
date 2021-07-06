# imports
from sklearn.metrics import classification_report
from data_loaders.table_data_loaders import get_dataloaders
from data_loaders.models import NeuralNet
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import matplotlib
from tableclass_engine_FFNN import train, validate, label_wise_metrics, \
    plot_loss_graph, plot_f1_graph, f1_nozeros
import numpy as np
import json
import os
from torch.utils.tensorboard import SummaryWriter

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
with open("../config/config_tableclass.json") as config:
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
all_train_loss = []
all_val_loss = []
all_train_f1_weighted = []
all_val_f1_weighted = []
all_train_f1_macro = []
all_val_f1_macro = []

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
    train_f1_positives_macro, train_f1_positives_weighted = f1_nozeros(train_class_report, remove_nonrel=True, only_notrel=False)
    val_f1_positives_macro, val_f1_positives_weighted = f1_nozeros(val_class_report, remove_nonrel=True, only_notrel=False)

    # update metrics super list
    all_train_loss.append(train_loss)
    all_val_loss.append(val_loss)
    all_train_f1_weighted.append(train_f1_positives_weighted)
    all_val_f1_weighted.append(val_f1_positives_weighted)
    all_train_f1_macro.append(train_f1_positives_macro)
    all_val_f1_macro.append(val_f1_positives_macro)

# plot and save the train and validation line graphs

plot_loss_graph(all_train_loss, all_val_loss, cf)
plot_f1_graph(all_train_f1_macro, all_val_f1_macro, cf, "- Positive Classes")
plot_f1_graph(all_train_f1_weighted, all_val_f1_weighted, cf, "- Weighted Positive Classes")


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
