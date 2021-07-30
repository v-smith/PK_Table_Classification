from data_loaders.table_data_loaders import get_dataloaders
import json

with open("../config/config_tableclass_FFNN.json") as config:
    cf = json.load(config)

train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset = get_dataloaders(
    inp_data_dir="../data/train-test-val",
    inp_tokenizer="../tokenizers/tokenizerPKtablesSpecialTokens5000.json",
    max_len=cf["max_len"], batch_size=cf["batch_size"], val_batch_size=cf["val_batch_size"],
    n_workers=cf["n_workers"], remove_html=cf["remove_html"], baseline_only=cf["baseline_only"],
    aug_all=False, aug_nums=False, aug_syns=False, aug_both=cf["aug_both"], sampler=cf["sampler"], sections=cf["sections_only"], multi_hot=cf["multi_hot"])

a= 1
