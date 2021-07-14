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
from prodigy.util import read_jsonl

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
    n_workers=cf["n_workers"],  baseline_only=False, remove_html=False)


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
    all_labs = []
    for i, batch in enumerate(test_dataloader):
        input_ids, target = batch['input_ids'].to(device), batch['labels']

        outputs = model(input_ids)
        outputs = torch.sigmoid(outputs).detach().cpu()
        predicted = torch.round(outputs)

        labels = np.asarray(target)
        all_labs.extend(labels)
        preds = np.asarray(predicted)
        all_preds.extend(preds)

# metric calculations
all_labels = np.stack(all_labs, axis=0)
all_predictions = np.stack(all_preds, axis=0)

test_class_dict = classification_report(all_labels,all_preds, output_dict=True)
test_class_report = classification_report(all_labels,all_preds)
test_f1_positives_macro, test_f1_positives_weighted = f1_nozeros(test_class_dict, remove_nonrel=True, only_notrel=False)

# prints
print(f"F1 Weighted (Positive Classes Only): {test_f1_positives_weighted} | "
      f"F1 not weighted (Macro and Positive Classes Only): {test_f1_positives_macro}")
print(f"Classification Report: {test_class_report}")
a = 1

#prepare prodigy dataset to compare labels and predicitions
all_tokens = [np.asarray(item["input_ids"]) for item in test_dataset]
htmls= [tokenizer.decode(tokens) for tokens in all_tokens]
task_hashes= [item["_task_hash"] for item in test_dataset]
input_hashes= [item["_input_hash"] for item in test_dataset]
binary_labels= [item for item in all_labs]
binary_preds= [item for item in all_preds]
indices_labels = [np.where(item == 1)[0].tolist() for item in binary_labels]
indices_preds = [np.where(item == 1)[0].tolist() for item in binary_preds]


compare_labels = []
compare_preds= []
for table, label, pred, t_hash, i_hash in zip(htmls, indices_labels, indices_preds, task_hashes, input_hashes):
    #if label != pred:
    compare_labels.append({"html": table, "accept": label, "_task_hash": t_hash, "_input_hash": i_hash, "_session_id": "labels", "answer": "accept", "_view_id":"blocks","options":[{"id":0,"text":"NC Params"},{"id":1,"text":"C Params"},{"id":2,"text":"Param-cov Rs"},{"id":3,"text":"Params Other"},{"id":4,"text":"Doses"},{"id":5,"text":"Number of Subjects"},{"id":6,"text":"Samples Timings"},{"id":7,"text":"Demographics"},{"id":8,"text":"Covariates Other"}], "config":{"choice_style":"multiple"}})
    compare_preds.append({"html": table, "accept": pred, "_task_hash": t_hash, "_input_hash": i_hash, "_session_id": "preds", "answer": "accept", "_view_id":"blocks","options":[{"id":0,"text":"NC Params"},{"id":1,"text":"C Params"},{"id":2,"text":"Param-cov Rs"},{"id":3,"text":"Params Other"},{"id":4,"text":"Doses"},{"id":5,"text":"Number of Subjects"},{"id":6,"text":"Samples Timings"},{"id":7,"text":"Demographics"},{"id":8,"text":"Covariates Other"}], "config":{"choice_style":"multiple"}})

original_annotations = list(read_jsonl("../data/train-test-val/test.jsonl"))

for d in compare_preds:
    for d2 in original_annotations:
        if d["_task_hash"] == d2["_task_hash"]:
            d["html"]= d2["html"]

for a in compare_labels:
    for a2 in original_annotations:
        if a["_task_hash"] == a2["_task_hash"]:
            a["html"]= a2["html"]

db = connect()
if db.get_dataset("labels"):
    db.drop_dataset("labels")
db.add_dataset("labels")
db.add_examples(compare_labels, ["labels"])
dataset_labs = db.get_dataset("labels")

if db.get_dataset("predictions"):
    db.drop_dataset("predictions")
db.add_dataset("predictions")
db.add_examples(compare_preds, ["predictions"])
dataset_preds = db.get_dataset("predictions")
print(f"===== length labels {len(dataset_labs)}, length of preds: {len(dataset_preds)}=====")

a=1
