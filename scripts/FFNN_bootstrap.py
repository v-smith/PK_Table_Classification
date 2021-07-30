# imports
import numpy
from sklearn.metrics import classification_report
from data_loaders.bootstrap_data_loaders import get_dataloaders, read_dataset
from data_loaders.models import NeuralNet
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import matplotlib
from tableclass_engine import validate, train, f1_nozeros, save_checkpoint
import numpy as np
import json
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json


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

#writer = SummaryWriter(log_dir=("../data/runs/" + cf["run_name"] + "bootstrap")) #, filename_suffix=cf["run_name"])

# ============ Load and Check Tokenizer =========== #
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = PreTrainedTokenizerFast(tokenizer_file=cf["tokenizer_file"])
#tokenizer.add_tokens(["[CAPTION]", "[FIRST_ROW]", "[FIRST_COL]", "[TABLE_BODY]"], special_tokens=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# get vocab size and padding index
vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens) #+ len(tokenizer.get_added_vocab())
padding_idx = tokenizer.pad_token_id

# ============ Set Device =============== #
# device config
device = 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.autograd.set_detect_anomaly(True)

# ============ Get Model =============== #
model = NeuralNet(num_classes=cf["num_classes"], embeds_size=cf["embeds_size"],
                  vocab_size=vocab_size, padding_idx=padding_idx, hidden_size=cf["hidden_size"], drop_out=cf["drop_out"]).to(device)

# ============ Define Loss and Optimiser =============== #
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cf["lr"])

#============ Bootstrap ===============#

#read in data
train_samples = list(read_dataset(data_dir_inp="../data/train-test-val", dataset_name="train-corrected"))
valid_samples = list(read_dataset(data_dir_inp="../data/train-test-val", dataset_name="val-corrected"))
train_samples.extend(valid_samples)
train_samples = [{"html": dic["html"], "accept": dic["accept"]} for dic in train_samples]
values = pd.DataFrame(train_samples).values

# configure bootstrap
n_iterations = 1
n_size = int(len(train_samples) * 0.50)

# run bootstrap
all_stats = []
best_stats = []

for i in tqdm(range(n_iterations)):
    print("\n ===============================")
    print(f"=====BOOTSTRAP ITERATION: {i} =====")
    print("===============================")
    # prepare train and test sets
    train_arr = resample(values, n_samples=n_size)
    test_arr = np.array([x for x in values if x.tolist() not in train_arr.tolist()])
    train_list = [x.tolist() for x in train_arr]
    test_list = [x.tolist() for x in test_arr]

    train_dataloader, train_dataset, valid_dataloader, valid_dataset = get_dataloaders(
        train_dataset=train_list, val_dataset=test_list,
        inp_tokenizer="../tokenizers/tokenizerPKtablesSpecialTokens5000.json",
        max_len=cf["max_len"], batch_size=cf["batch_size"], val_batch_size=cf["val_batch_size"],
        n_workers=cf["n_workers"], remove_html=cf["remove_html"], baseline_only=cf["baseline_only"],
        aug_all=cf["aug_all"], aug_nums=cf["aug_nums"], aug_syns=cf["aug_syns"], aug_both= cf["aug_both"], sampler=cf["sampler"],
        sections=cf["sections_only"],  multi_hot=cf["multi_hot"])

    print("\n ===============================")
    print(f"=====BOOTSTRAP ITERATION: {i} =====")
    print("===============================")

    # ============ Train and Val Loop  =============== #
    epochs = cf["epochs"]
    all_train_loss = []
    all_val_loss = []
    all_train_f1_weighted = []
    all_val_f1_weighted = []
    all_train_f1_macro = []
    all_val_f1_macro = []

    for epoch in range(epochs):
        print("\n ===============================")
        print(f"=====BOOTSTRAP ITERATION: {i} =====")
        print("===============================")
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
        #writer.add_scalar("Loss/train", train_loss, epoch+1)
        #writer.add_scalar("Loss/val", val_loss, epoch+1)
        #writer.add_scalar("F1_weighted/train", train_f1_positives_weighted, epoch+1)
        #writer.add_scalar("F1_weighted/val", val_f1_positives_weighted, epoch+1)

        # update metrics super list
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_f1_weighted.append(train_f1_positives_weighted)
        all_val_f1_weighted.append(val_f1_positives_weighted)
        all_train_f1_macro.append(train_f1_positives_macro)
        all_val_f1_macro.append(val_f1_positives_macro)

    #final scores
    all_stats.append(all_val_f1_weighted)

#make sure that all pending events have been written to disk
#writer.flush()
#close writer
#writer.close()

max_stats = [max(lst) for lst in all_stats]
plt.hist(max_stats)
plt.show()

#confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2) *100
lower = max(0.0, numpy.percentile(max_stats, p))
p = (alpha+((1.0-alpha)/2.0))*100
upper = min(1.0, numpy.percentile(max_stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))

with open("../data/outputs/bootstrap/bootstrap.json") as feedsjson:
    feeds = json.load(feedsjson)
    name = str(cf["run_name"]) + str(n_iterations)
    entry = {name: max_stats, "upper": upper, "lower": lower}
    feeds.append(entry)
with open("../data/outputs/bootstrap/bootstrap.json", mode='w') as f:
    f.write(json.dumps(feeds, indent=2))


a=1
